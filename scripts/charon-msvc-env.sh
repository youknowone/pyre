# shellcheck shell=bash
# Prepend the MSVC linker directory to PATH on a Windows host.
#
# Git for Windows ships a GNU coreutils `link.exe` in /usr/bin that shadows
# the MSVC `link.exe` rustc needs to link host build scripts. Without this,
# building Charon or extracting a crate that compiles fresh build-deps (e.g.
# cranelift's libm / zerocopy) fails with "/usr/bin/link: extra operand".
# Locate the MSVC tools via vswhere and put them first. No-op off Windows.
charon_prepend_msvc_link() {
    case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ;; *) return 0 ;; esac
    local vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
    if [[ ! -x "$vswhere" ]]; then
        echo "warn: vswhere not found; MSVC link.exe may be shadowed by Git's link.exe" >&2
        return 0
    fi
    local install link
    install="$("$vswhere" -latest -products '*' \
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
        -property installationPath 2>/dev/null | tr -d '\r')"
    [[ -n "$install" ]] || return 0
    # vswhere returns a Windows path (`Z:\...`); convert to POSIX for git-bash.
    install="$(cygpath -u "$install" 2>/dev/null || echo "$install")"
    # Pick the highest installed MSVC toolset version. `-ipath` because the
    # on-disk directory case (Hostx64) varies and glob matching is case-sensitive.
    link="$(find "$install/VC/Tools/MSVC" -ipath '*/hostx64/x64/link.exe' 2>/dev/null | sort | tail -1)"
    if [[ -n "$link" ]]; then
        export PATH="$(dirname "$link"):$PATH"
    else
        echo "warn: MSVC link.exe not found under $install; Git's link.exe may shadow it" >&2
    fi
}
