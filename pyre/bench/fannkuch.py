def fannkuch(n):
    perm1 = [0] * n
    count = [0] * n
    perm = [0] * n
    i = 0
    while i < n:
        perm1[i] = i
        i = i + 1
    maxflips = 0
    checksum = 0
    nperm = 0
    r = n

    while True:
        while r > 1:
            count[r - 1] = r
            r = r - 1

        if perm1[0] != 0 and perm1[n - 1] != n - 1:
            # copy perm1 to perm
            i = 0
            while i < n:
                perm[i] = perm1[i]
                i = i + 1
            flips = 0
            k = perm[0]
            while k != 0:
                i = 0
                j = k
                while i < j:
                    t = perm[i]
                    perm[i] = perm[j]
                    perm[j] = t
                    i = i + 1
                    j = j - 1
                flips = flips + 1
                k = perm[0]
            if flips > maxflips:
                maxflips = flips
            if nperm % 2 == 0:
                checksum = checksum + flips
            else:
                checksum = checksum - flips

        nperm = nperm + 1

        while True:
            if r == n:
                print(checksum)
                print(maxflips)
                return maxflips
            p0 = perm1[0]
            i = 0
            while i < r:
                perm1[i] = perm1[i + 1]
                i = i + 1
            perm1[r] = p0
            count[r] = count[r] - 1
            if count[r] > 0:
                r = 1
                break
            r = r + 1

fannkuch(10)
