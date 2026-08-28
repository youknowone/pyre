# pyre-check: gate=1
# pyre-check: platforms=darwin
# CPython-suite gap: the suite reads these constants only through the calls
# that take them, and every such test skips when the name is absent, so a
# missing name is silently green.
# parity-tests reason: `socketmodule.c`, `mmapmodule.c`, `fcntlmodule.c`,
# `posixmodule.c`, `signalmodule.c`, `syslogmodule.c`, `termios.c` and
# `timemodule.c` publish every name the header defines, and 110 of darwin's
# were absent here while `socket.IP_MAX_MEMBERSHIPS` carried linux's 20.

# pyre-check: pypy-diverges: `platform.DefinedConstantInteger` reaches only the
# names each `*_rffi.py` lists, and the darwin-only half of these is not in
# those lists.

# Every value below was read back from the SDK headers, not from an
# interpreter: `<sys/fcntl.h>`, `<sys/mman.h>`, `<sys/resource.h>`,
# `<copyfile.h>`, `<signal.h>`, `<sys/socket.h>`, `<netinet/in.h>`,
# `<netinet/tcp.h>`, `<net/ethernet.h>`, `<sys/sys_domain.h>`, `<syslog.h>`,
# `<sys/termios.h>` and `<time.h>`.

import importlib

EXPECTED = {
    "fcntl": {
        "FASYNC": 64,
        "F_FULLFSYNC": 51,
        "F_GETLEASE": 107,
        "F_GETNOSIGPIPE": 74,
        "F_GETPATH": 50,
        "F_NOCACHE": 48,
        "F_OFD_GETLK": 92,
        "F_OFD_SETLK": 90,
        "F_OFD_SETLKW": 91,
        "F_RDAHEAD": 45,
        "F_SETLEASE": 106,
        "F_SETNOSIGPIPE": 73,
    },
    "mmap": {
        "MADV_FREE": 5,
        "MADV_FREE_REUSABLE": 7,
        "MADV_FREE_REUSE": 8,
        "MAP_32BIT": 32768,
        "MAP_HASSEMAPHORE": 512,
        "MAP_JIT": 2048,
        "MAP_NOCACHE": 1024,
        "MAP_NOEXTEND": 256,
        "MAP_NORESERVE": 64,
        "MAP_RESILIENT_CODESIGN": 8192,
        "MAP_RESILIENT_MEDIA": 16384,
        "MAP_TPRO": 524288,
        "MAP_TRANSLATED_ALLOW_EXECUTE": 131072,
        "MAP_UNIX03": 262144,
    },
    "posix": {
        "NGROUPS_MAX": 16,
        "PRIO_DARWIN_BG": 4096,
        "PRIO_DARWIN_NONUI": 4097,
        "PRIO_DARWIN_PROCESS": 4,
        "PRIO_DARWIN_THREAD": 3,
        "TMP_MAX": 308915776,
        "_COPYFILE_ACL": 1,
        "_COPYFILE_DATA": 8,
        "_COPYFILE_STAT": 2,
        "_COPYFILE_XATTR": 4,
    },
    "signal": {
        "SIGEMT": 7,
        "SIGINFO": 29,
        "SIGIOT": 6,
    },
    "socket": {
        "AF_APPLETALK": 16,
        "AF_DECnet": 12,
        "AF_IPX": 23,
        "AF_LINK": 18,
        "AF_SNA": 11,
        "AF_SYSTEM": 32,
        "ETHERTYPE_ARP": 2054,
        "ETHERTYPE_IP": 2048,
        "ETHERTYPE_IPV6": 34525,
        "ETHERTYPE_VLAN": 33024,
        "IPPROTO_EON": 80,
        "IPPROTO_GGP": 3,
        "IPPROTO_HELLO": 63,
        "IPPROTO_IPCOMP": 108,
        "IPPROTO_IPV4": 4,
        "IPPROTO_ND": 77,
        "IPPROTO_XTP": 36,
        "IPV6_DONTFRAG": 62,
        "IPV6_DSTOPTS": 50,
        "IPV6_HOPOPTS": 49,
        "IPV6_NEXTHOP": 48,
        "IPV6_PATHMTU": 44,
        "IPV6_RECVDSTOPTS": 40,
        "IPV6_RECVHOPOPTS": 39,
        "IPV6_RECVPATHMTU": 43,
        "IPV6_RECVRTHDR": 38,
        "IPV6_RTHDR": 51,
        "IPV6_RTHDRDSTOPTS": 57,
        "IPV6_RTHDR_TYPE_0": 0,
        "IPV6_USE_MIN_MTU": 42,
        "IP_ADD_SOURCE_MEMBERSHIP": 70,
        "IP_BLOCK_SOURCE": 72,
        "IP_DROP_SOURCE_MEMBERSHIP": 71,
        "IP_MAX_MEMBERSHIPS": 4095,
        "IP_OPTIONS": 1,
        "IP_PKTINFO": 26,
        "IP_RECVDSTADDR": 7,
        "IP_RECVOPTS": 5,
        "IP_RECVRETOPTS": 6,
        "IP_RECVTOS": 27,
        "IP_RECVTTL": 24,
        "IP_RETOPTS": 8,
        "IP_UNBLOCK_SOURCE": 73,
        "LOCAL_PEERCRED": 1,
        "MSG_EOF": 256,
        "MSG_NOSIGNAL": 524288,
        "PF_SYSTEM": 32,
        "SCM_CREDS": 3,
        "SO_BINDTODEVICE": 4404,
        "SO_USELOOPBACK": 64,
        "SYSPROTO_CONTROL": 2,
        "TCP_CONNECTION_INFO": 262,
        "TCP_FASTOPEN": 261,
        "TCP_KEEPCNT": 258,
        "TCP_KEEPINTVL": 257,
        "TCP_NOTSENT_LOWAT": 513,
    },
    "syslog": {
        "LOG_AUTHPRIV": 80,
        "LOG_FTP": 88,
        "LOG_INSTALL": 112,
        "LOG_LAUNCHD": 192,
        "LOG_NETINFO": 96,
        "LOG_ODELAY": 4,
        "LOG_RAS": 120,
        "LOG_REMOTEAUTH": 104,
    },
    "termios": {
        "_POSIX_VDISABLE": 255,
    },
    "time": {
        "CLOCK_MONOTONIC_RAW": 4,
        "CLOCK_MONOTONIC_RAW_APPROX": 5,
        "CLOCK_UPTIME_RAW": 8,
        "CLOCK_UPTIME_RAW_APPROX": 9,
    },
}

for module_name, constants in EXPECTED.items():
    module = importlib.import_module(module_name)
    for name, value in constants.items():
        assert hasattr(module, name), f"{module_name}.{name} is missing"
        assert getattr(module, name) == value, (
            f"{module_name}.{name} is {getattr(module, name)}, expected {value}"
        )

# `mmapmodule.c` publishes `PROT_READ`/`PROT_WRITE`/`PROT_EXEC` and no
# `PROT_NONE`, so a module that answers to it is one name wide.
import mmap

assert not hasattr(mmap, "PROT_NONE")

print("OK")
