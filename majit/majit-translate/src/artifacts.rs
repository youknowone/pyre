//! Build-process transport for `CodeWriter.make_jitcodes` and `Assembler` tables.
//!
//! RPython's translator retains these objects in process. Rust build scripts
//! serialize them for the final binary; consumers supply paths and runtime
//! symbol bindings, while this module owns the wire format and validation.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::jitcode::{BhDescr, JitCode};

/// Dense allocation order, with a separate graph key for colliding leaf names.
#[derive(Serialize, Deserialize)]
pub struct JitCodeIndex {
    pub names: Vec<String>,
    pub paths: Vec<String>,
    pub offsets: Vec<u32>,
}

impl JitCodeIndex {
    pub fn encode(
        jitcodes: &[Arc<JitCode>],
        paths: Vec<String>,
    ) -> bincode::Result<(Self, Vec<u8>)> {
        assert_eq!(paths.len(), jitcodes.len());
        let mut bodies = Vec::new();
        let mut index = Self {
            names: Vec::new(),
            paths,
            offsets: vec![0],
        };
        for jitcode in jitcodes {
            index.names.push(jitcode.name.clone());
            bincode::serialize_into(&mut bodies, jitcode)?;
            index
                .offsets
                .push(u32::try_from(bodies.len()).expect("JitCode archive exceeds u32 offsets"));
        }
        Ok((index, bodies))
    }

    pub fn decode(bytes: &[u8], bodies: &[u8]) -> bincode::Result<Self> {
        let index: Self = bincode::deserialize(bytes)?;
        index.validate(bodies)?;
        Ok(index)
    }

    fn validate(&self, bodies: &[u8]) -> bincode::Result<()> {
        if self.paths.len() != self.names.len()
            || self.offsets.len() != self.names.len() + 1
            || self.offsets.first().copied() != Some(0)
            || self.offsets.last().map(|&n| n as usize) != Some(bodies.len())
            || !self.offsets.windows(2).all(|p| p[0] <= p[1])
        {
            return Err(Box::new(bincode::ErrorKind::Custom(
                "invalid JitCode archive index".into(),
            )));
        }
        Ok(())
    }

    pub fn load(&self, bodies: &[u8], index: usize) -> bincode::Result<Arc<JitCode>> {
        let start = self.offsets.get(index).copied();
        let end = index
            .checked_add(1)
            .and_then(|i| self.offsets.get(i))
            .copied();
        let bytes = start
            .zip(end)
            .and_then(|(s, e)| bodies.get(s as usize..e as usize))
            .ok_or_else(|| {
                Box::new(bincode::ErrorKind::Custom(
                    "JitCode index out of bounds".into(),
                ))
            })?;
        bincode::deserialize(bytes)
    }
}

/// Encoded bodies remain bytes until requested, so this envelope is Send+Sync
/// without publishing translation-time graphs across threads.
#[derive(Serialize, Deserialize)]
pub struct EmbeddedArtifacts {
    version: u32,
    pub index: JitCodeIndex,
    bodies: Vec<u8>,
    descrs: Vec<u8>,
    pub symbolic_fnaddrs: Vec<(i64, String)>,
    pub liveness: Vec<u8>,
}

impl EmbeddedArtifacts {
    pub fn from_pipeline(
        pipeline: &crate::pipeline::ProgramPipelineResult,
    ) -> bincode::Result<Self> {
        let mut paths = vec![String::new(); pipeline.jitcodes.len()];
        for (path, jitcode) in &pipeline.jitcodes_by_path {
            paths[jitcode.index()] = path.canonical_key();
        }
        let (index, bodies) = JitCodeIndex::encode(&pipeline.jitcodes, paths)?;
        Ok(Self {
            version: 1,
            index,
            bodies,
            descrs: bincode::serialize(&pipeline.descrs)?,
            symbolic_fnaddrs: pipeline.symbolic_fnaddr_paths.clone(),
            liveness: pipeline.all_liveness.clone(),
        })
    }

    pub fn encode(&self) -> bincode::Result<Vec<u8>> {
        bincode::serialize(self)
    }

    pub fn decode(bytes: &[u8]) -> bincode::Result<Self> {
        let artifacts: Self = bincode::deserialize(bytes)?;
        if artifacts.version != 1 {
            return Err(Box::new(bincode::ErrorKind::Custom(
                "unsupported JitCode artifact version".into(),
            )));
        }
        artifacts.index.validate(&artifacts.bodies)?;
        Ok(artifacts)
    }

    pub fn jitcodes(&self) -> bincode::Result<Vec<Arc<JitCode>>> {
        (0..self.index.names.len())
            .map(|i| self.index.load(&self.bodies, i))
            .collect()
    }

    pub fn descrs(&self) -> bincode::Result<Vec<BhDescr>> {
        bincode::deserialize(&self.descrs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_format_preserves_existing_tuple_wire_format() {
        let old = bincode::serialize(&(
            vec!["same".to_string(); 2],
            vec!["a::same".to_string(), "b::same".to_string()],
            vec![0_u32, 1, 2],
        ))
        .unwrap();
        let index = JitCodeIndex::decode(&old, &[0, 0]).unwrap();
        assert_eq!(index.paths, ["a::same", "b::same"]);
        assert_eq!(bincode::serialize(&index).unwrap(), old);
    }

    #[test]
    fn rejects_corrupt_offsets_and_lengths() {
        for offsets in [vec![], vec![1_u32, 2], vec![0, 3], vec![0, 2, 1]] {
            let bytes =
                bincode::serialize(&(vec!["one".to_string()], vec![String::new()], offsets))
                    .unwrap();
            assert!(JitCodeIndex::decode(&bytes, &[0, 0]).is_err());
        }
    }

    #[test]
    fn envelope_roundtrip_preserves_bodies_and_side_tables() {
        let code = Arc::new(JitCode::new("helper"));
        code.set_index(0);
        code.set_body(crate::jitcode::JitCodeBody {
            constants_i: vec![17, -4],
            ..Default::default()
        });
        let (index, bodies) = JitCodeIndex::encode(&[code], vec!["module::helper".into()]).unwrap();
        let mut original = EmbeddedArtifacts {
            version: 1,
            index,
            bodies,
            descrs: bincode::serialize(&Vec::<BhDescr>::new()).unwrap(),
            symbolic_fnaddrs: vec![(17, "runtime::helper".into())],
            liveness: vec![1, 2, 3],
        };
        let loaded = EmbeddedArtifacts::decode(&original.encode().unwrap()).unwrap();
        let codes = loaded.jitcodes().unwrap();
        assert_eq!(codes[0].name, "helper");
        assert_eq!(codes[0].index(), 0);
        assert_eq!(codes[0].body().constants_i, [17, -4]);
        assert_eq!(loaded.index.paths, ["module::helper"]);
        assert_eq!(loaded.symbolic_fnaddrs, original.symbolic_fnaddrs);
        assert_eq!(loaded.liveness, original.liveness);
        assert!(loaded.descrs().unwrap().is_empty());
        assert!(loaded.index.load(&loaded.bodies, 1).is_err());
        assert!(loaded.index.load(&loaded.bodies, usize::MAX).is_err());
        original.version = 2;
        assert!(EmbeddedArtifacts::decode(&original.encode().unwrap()).is_err());
    }
}
