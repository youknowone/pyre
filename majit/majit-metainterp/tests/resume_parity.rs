use majit_metainterp::resume::{
    FrameInfo, FrameSlotSource, ReconstructedValue, ResumeData, VirtualFieldSource, VirtualInfo,
};

const TAG_CONST: i64 = 0;
const TAG_INT: i64 = 1;
const TAG_BOX: i64 = 2;
const TAG_VIRTUAL: i64 = 3;
const ENCODED_UNINITIALIZED: i64 = -2;
const ENCODED_UNAVAILABLE: i64 = -3;

fn untag(word: i64) -> (i64, i64) {
    (word >> 2, word & 0b11)
}

#[test]
fn resume_py_public_encoding_uses_tagged_numbering() {
    let large_const = (1_i64 << 62) + 9;
    let rd = ResumeData {
        frames: vec![FrameInfo {
            pc: 123,
            slot_map: vec![
                FrameSlotSource::FailArg(2),
                FrameSlotSource::Constant(7),
                FrameSlotSource::Constant(large_const),
                FrameSlotSource::Virtual(0),
                FrameSlotSource::Uninitialized,
                FrameSlotSource::Unavailable,
            ],
        }],
        virtuals: vec![VirtualInfo::VArray {
            descr_index: 9,
            items: vec![VirtualFieldSource::Constant(large_const)],
        }],
        pending_fields: Vec::new(),
    };

    let encoded = rd.encode();
    assert_eq!(encoded.code[0] as usize, encoded.code.len());
    assert_eq!(encoded.num_fail_args, 1);
    assert_eq!(encoded.fail_arg_positions, vec![2]);
    assert_eq!(encoded.consts, vec![large_const]);

    let slot_words = &encoded.code[5..11];
    assert_eq!(untag(slot_words[0]), (0, TAG_BOX));
    assert_eq!(untag(slot_words[1]), (7, TAG_INT));
    assert_eq!(untag(slot_words[2]), (0, TAG_CONST));
    assert_eq!(untag(slot_words[3]), (0, TAG_VIRTUAL));
    assert_eq!(untag(slot_words[4]), (ENCODED_UNINITIALIZED, TAG_CONST));
    assert_eq!(untag(slot_words[5]), (ENCODED_UNAVAILABLE, TAG_CONST));
}

#[test]
fn resume_py_public_roundtrip_recovers_virtualized_state() {
    let rd = ResumeData {
        frames: vec![FrameInfo {
            pc: 77,
            slot_map: vec![
                FrameSlotSource::FailArg(0),
                FrameSlotSource::Constant(42),
                FrameSlotSource::Virtual(0),
                FrameSlotSource::Unavailable,
            ],
        }],
        virtuals: vec![VirtualInfo::VirtualObj {
            type_id: 1,
            descr_index: 3,
            fields: vec![
                (0, VirtualFieldSource::FailArg(1)),
                (1, VirtualFieldSource::Constant(99)),
            ],
            fielddescrs: vec![],
            descr_size: 0,
        }],
        pending_fields: vec![majit_metainterp::resume::PendingFieldInfo {
            descr_index: 5,
            target: FrameSlotSource::FailArg(0),
            value: FrameSlotSource::Constant(123),
            item_index: Some(2),
        }],
    };

    let state = rd.reconstruct_state(&[7, 88]);
    assert_eq!(state.frames.len(), 1);
    assert_eq!(
        state.frames[0].values,
        vec![
            ReconstructedValue::Value(7),
            ReconstructedValue::Value(42),
            ReconstructedValue::Virtual(0),
            ReconstructedValue::Unavailable,
        ]
    );
    assert_eq!(state.virtuals.len(), 1);
    assert_eq!(
        state.pending_fields,
        vec![majit_metainterp::resume::ResolvedPendingFieldWrite {
            descr_index: 5,
            target: majit_metainterp::resume::MaterializedValue::Value(7),
            value: majit_metainterp::resume::MaterializedValue::Value(123),
            item_index: Some(2),
        }]
    );
    assert_eq!(
        state.virtuals[0],
        majit_metainterp::resume::MaterializedVirtual::Obj {
            type_id: 1,
            descr_index: 3,
            fields: vec![
                (0, majit_metainterp::resume::MaterializedValue::Value(88)),
                (1, majit_metainterp::resume::MaterializedValue::Value(99)),
            ],
        }
    );
}
