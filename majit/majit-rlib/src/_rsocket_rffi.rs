//! `rpython/rlib/_rsocket_rffi.py` — the Winsock last-error accessor.

#[link(name = "ws2_32")]
unsafe extern "system" {
    #[link_name = "WSAGetLastError"]
    pub fn _WSAGetLastError() -> i32;
}
