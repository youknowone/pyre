use blake2::{Blake2b512, Blake2s256};
use md5::Md5;
use sha1::Sha1;
use sha2::{Digest, Sha224, Sha256, Sha384, Sha512};
use sha3::{
    Sha3_224, Sha3_256, Sha3_384, Sha3_512, Shake128, Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};

#[inline(never)]
pub fn compute_digest(name: &str, data: &[u8], length: usize) -> Option<Vec<u8>> {
    let digest = match name {
        "md5" => Md5::digest(data).to_vec(),
        "sha1" => Sha1::digest(data).to_vec(),
        "sha224" => Sha224::digest(data).to_vec(),
        "sha256" => Sha256::digest(data).to_vec(),
        "sha384" => Sha384::digest(data).to_vec(),
        "sha512" => Sha512::digest(data).to_vec(),
        "sha3_224" => Sha3_224::digest(data).to_vec(),
        "sha3_256" => Sha3_256::digest(data).to_vec(),
        "sha3_384" => Sha3_384::digest(data).to_vec(),
        "sha3_512" => Sha3_512::digest(data).to_vec(),
        "blake2b" => Blake2b512::digest(data).to_vec(),
        "blake2s" => Blake2s256::digest(data).to_vec(),
        "shake_128" => {
            let mut h = Shake128::default();
            h.update(data);
            let mut out = vec![0u8; length];
            h.finalize_xof().read(&mut out);
            out
        }
        "shake_256" => {
            let mut h = Shake256::default();
            h.update(data);
            let mut out = vec![0u8; length];
            h.finalize_xof().read(&mut out);
            out
        }
        _ => return None,
    };
    Some(digest)
}

#[cfg(test)]
mod tests {
    use super::compute_digest;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn computes_fixed_length_digests() {
        assert_eq!(
            hex(&compute_digest("md5", b"abc", 0).unwrap()),
            "900150983cd24fb0d6963f7d28e17f72"
        );
        assert_eq!(
            hex(&compute_digest("sha256", b"abc", 0).unwrap()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn computes_extendable_output_digests() {
        let digest = compute_digest("shake_128", b"abc", 8).unwrap();
        assert_eq!(digest.len(), 8);
        assert_eq!(hex(&digest), "5881092dd818bf5c");
    }

    #[test]
    fn rejects_unknown_algorithm() {
        assert!(compute_digest("not-a-hash", b"abc", 0).is_none());
    }
}
