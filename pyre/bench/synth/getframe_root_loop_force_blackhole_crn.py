# Regression guard: SIGSEGV from a NULL operand-stack slot written by the
# single-frame blackhole's ContinueRunningNormally handoff.
#
# The walk roots at `main`, which HAS a Python loop, so its portal jitcode
# carries a jit_merge_point at the loop header. A sys._getframe force inside the
# loop latched a blackhole image built at the resume pc just past the residual;
# driving it ran to the loop back edge and raised ContinueRunningNormally at the
# merge point. The MIFrame is seeded from the live colors at the BUILD pc, but
# the merge point has its own live set, so a Ref color live at the merge but not
# at the build read back NULL -- and apply_blackhole_crn had no NULL guard, so it
# wrote a null into a live operand-stack slot. Resuming there faulted the
# interpreter (EXC_BAD_ACCESS at 0x0 in baseobjspace::next), exit 139, JIT-only.
#
# The walk's own flush declines exactly this case ("NULL operand-stack shadow
# slot (mid-expression)"); the blackhole path did not.
#
# Guarding it post-drive is NOT sufficient and this fixture also pins that: a
# post-drive decline discards a region the drive already executed and hands it
# back to the replay, which turned 199990000 into 200005595. The gate has to be
# taken before the drive.
import sys


def main():
    total = 0
    names = set()
    for i in range(20000):
        fr = sys._getframe(0)
        names.add(fr.f_code.co_name)
        total += i
    print(total, sorted(names))


main()
