//! Shared digest engine, kept outside interpreter LLBC extraction.
//!
//! PyPy's lib_pypy._hashlib.HASH owns the Python adapter; rustpython-common
//! supplies the VM-independent digest operations behind that boundary.
pub use rustpython_common::hashlib::*;

/// Extra words for aligning a pointer-word-aligned Python payload to the
/// engine's alignment, including on wasm32 where a word is only four bytes.
pub const STATE_STORAGE_ALIGN_SLACK_WORDS: usize =
    (HASH_STATE_STORAGE_ALIGN - std::mem::size_of::<usize>()) / std::mem::size_of::<usize>();

#[cfg(test)]
mod tests {
    use super::{
        HASH_STATE_STORAGE_WORDS, HMAC_STATE_STORAGE_WORDS, STATE_STORAGE_ALIGN_SLACK_WORDS,
        compute_digest, compute_pbkdf2_hmac, compute_scrypt, hmac_state_digest, hmac_state_drop,
        hmac_state_init, hmac_state_update, state_copy, state_digest, state_drop, state_init,
        state_init_blake2, state_update,
    };

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    struct HashStorage([usize; HASH_STATE_STORAGE_WORDS + STATE_STORAGE_ALIGN_SLACK_WORDS]);

    struct HmacStorage([usize; HMAC_STATE_STORAGE_WORDS + STATE_STORAGE_ALIGN_SLACK_WORDS]);

    fn aligned_ptr(words: &[usize]) -> *const usize {
        let address = words.as_ptr() as usize;
        ((address + 15) & !15) as *const usize
    }

    fn aligned_mut_ptr(words: &mut [usize]) -> *mut usize {
        aligned_ptr(words) as *mut usize
    }

    #[test]
    fn shared_engine_matches_pypy_digest_vectors() {
        // Oracle: lib_pypy._hashlib.HASH, hashlib.new(name, b"abc").
        for (name, expected) in [
            ("md5", "900150983cd24fb0d6963f7d28e17f72"),
            ("sha1", "a9993e364706816aba3e25717850c26c9cd0d89d"),
            (
                "sha224",
                "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7",
            ),
            (
                "sha256",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                "sha384",
                "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7",
            ),
            (
                "sha512",
                "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
            ),
            (
                "sha3_224",
                "e642824c3f8cf24ad09234ee7d3c766fc9a3a5168d0c94ad73b46fdf",
            ),
            (
                "sha3_256",
                "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532",
            ),
            (
                "sha3_384",
                "ec01498288516fc926459f58e2c6ad8df9b473cb0fc08c2596da7cf0e49be4b298d88cea927ac7f539f1edf228376d25",
            ),
            (
                "sha3_512",
                "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0",
            ),
            (
                "blake2b",
                "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923",
            ),
            (
                "blake2s",
                "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982",
            ),
        ] {
            assert_eq!(
                hex(&compute_digest(name, b"abc", 0).unwrap()),
                expected,
                "{name}"
            );
            let mut state =
                HashStorage([0; HASH_STATE_STORAGE_WORDS + STATE_STORAGE_ALIGN_SLACK_WORDS]);
            unsafe {
                let storage = aligned_mut_ptr(&mut state.0);
                assert!(state_init(storage, HASH_STATE_STORAGE_WORDS, name));
                state_update(storage, HASH_STATE_STORAGE_WORDS, b"a");
                state_update(storage, HASH_STATE_STORAGE_WORDS, b"bc");
                let digest = state_digest(storage, HASH_STATE_STORAGE_WORDS, 0);
                state_drop(storage, HASH_STATE_STORAGE_WORDS);
                assert_eq!(hex(&digest), expected, "{name}");
            }
        }
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

    #[test]
    fn incremental_state_updates_and_copies_independently() {
        let mut state =
            HashStorage([0usize; HASH_STATE_STORAGE_WORDS + STATE_STORAGE_ALIGN_SLACK_WORDS]);
        let mut clone =
            HashStorage([0usize; HASH_STATE_STORAGE_WORDS + STATE_STORAGE_ALIGN_SLACK_WORDS]);
        unsafe {
            assert!(state_init(
                aligned_mut_ptr(&mut state.0),
                HASH_STATE_STORAGE_WORDS,
                "sha256"
            ));
            state_update(
                aligned_mut_ptr(&mut state.0),
                HASH_STATE_STORAGE_WORDS,
                b"ab",
            );
            state_copy(
                aligned_ptr(&state.0),
                aligned_mut_ptr(&mut clone.0),
                HASH_STATE_STORAGE_WORDS,
            );
            state_update(
                aligned_mut_ptr(&mut state.0),
                HASH_STATE_STORAGE_WORDS,
                b"c",
            );
            state_update(
                aligned_mut_ptr(&mut clone.0),
                HASH_STATE_STORAGE_WORDS,
                b"d",
            );
            assert_eq!(
                state_digest(aligned_ptr(&state.0), HASH_STATE_STORAGE_WORDS, 0),
                compute_digest("sha256", b"abc", 0).unwrap()
            );
            assert_eq!(
                state_digest(aligned_ptr(&clone.0), HASH_STATE_STORAGE_WORDS, 0),
                compute_digest("sha256", b"abd", 0).unwrap()
            );
            state_drop(aligned_mut_ptr(&mut state.0), HASH_STATE_STORAGE_WORDS);
            state_drop(aligned_mut_ptr(&mut clone.0), HASH_STATE_STORAGE_WORDS);
        }
    }

    #[test]
    fn incremental_hmac_matches_rfc_4231_sha256() {
        let mut state =
            HmacStorage([0usize; HMAC_STATE_STORAGE_WORDS + STATE_STORAGE_ALIGN_SLACK_WORDS]);
        unsafe {
            assert!(hmac_state_init(
                aligned_mut_ptr(&mut state.0),
                HMAC_STATE_STORAGE_WORDS,
                "sha256",
                &[0x0b; 20],
            ));
            hmac_state_update(
                aligned_mut_ptr(&mut state.0),
                HMAC_STATE_STORAGE_WORDS,
                b"Hi There",
            );
            assert_eq!(
                hex(&hmac_state_digest(
                    aligned_ptr(&state.0),
                    HMAC_STATE_STORAGE_WORDS,
                )),
                "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
            );
            hmac_state_drop(aligned_mut_ptr(&mut state.0), HMAC_STATE_STORAGE_WORDS);
        }
    }

    #[test]
    fn pbkdf2_hmac_matches_rfc_6070_sha1() {
        assert_eq!(
            hex(&compute_pbkdf2_hmac("sha1", b"password", b"salt", 2, 20).unwrap()),
            "ea6c014dc72d6f8ccd1ed92ace1d41f0d8de8957"
        );
    }

    #[test]
    fn scrypt_matches_rfc_7914() {
        assert_eq!(
            hex(&compute_scrypt(b"", b"", 4, 1, 1, 64).unwrap()),
            "77d6576238657b203b19ca42c18a0497f16b4844e3074ae8dfdffa3fede21442\
             fcd0069ded0948f8326a753a0fc81f17e8d3e0fb2e0d3628cf35e20c38d18906"
                .replace(' ', "")
        );
    }

    #[test]
    fn blake2_parameter_block_matches_cpython_vectors() {
        for (name, expected) in [
            ("blake2b", "920568b0c5873b2f0ab67bedb6cf1b2b"),
            ("blake2s", "bf2a8f7fe3c555012a6f8046e646bc75"),
        ] {
            let mut state =
                HashStorage([0usize; HASH_STATE_STORAGE_WORDS + STATE_STORAGE_ALIGN_SLACK_WORDS]);
            unsafe {
                assert!(state_init_blake2(
                    aligned_mut_ptr(&mut state.0),
                    HASH_STATE_STORAGE_WORDS,
                    name,
                    16,
                    b"bar",
                    b"baz",
                    b"bing",
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    true,
                ));
                state_update(
                    aligned_mut_ptr(&mut state.0),
                    HASH_STATE_STORAGE_WORDS,
                    b"foo",
                );
                assert_eq!(
                    hex(&state_digest(
                        aligned_ptr(&state.0),
                        HASH_STATE_STORAGE_WORDS,
                        0,
                    )),
                    expected
                );
                state_drop(aligned_mut_ptr(&mut state.0), HASH_STATE_STORAGE_WORDS);
            }
        }
    }
}
