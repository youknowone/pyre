PI = 3.141592653589793
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24

def make_body(x, y, z, vx, vy, vz, mass):
    return [x, y, z, vx, vy, vz, mass]

def make_sun():
    return make_body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS)

def make_jupiter():
    return make_body(
        4.84143144246472090,
        -1.16032004402742839,
        -1.03622044471123109,
        0.00166007664274403694 * DAYS_PER_YEAR,
        0.00769901118419740425 * DAYS_PER_YEAR,
        -0.00690460016972063023 * DAYS_PER_YEAR,
        0.000954791938424326609 * SOLAR_MASS)

def make_saturn():
    return make_body(
        8.34336671824457987,
        4.12479856412430479,
        -4.03523417114321381,
        -0.00276742510726862411 * DAYS_PER_YEAR,
        0.00499852801234917238 * DAYS_PER_YEAR,
        0.00230417297573763929 * DAYS_PER_YEAR,
        0.000285885980666130812 * SOLAR_MASS)

def make_uranus():
    return make_body(
        12.8943695621391310,
        -15.1111514016986312,
        -0.223307578892655734,
        0.00296460137564761618 * DAYS_PER_YEAR,
        0.00237847173959480950 * DAYS_PER_YEAR,
        -0.00029658956854023756 * DAYS_PER_YEAR,
        0.0000436624404335156298 * SOLAR_MASS)

def make_neptune():
    return make_body(
        15.3796971148509165,
        -25.9193146099879641,
        0.179258772950371181,
        0.00268067772490389322 * DAYS_PER_YEAR,
        0.00162824170038242295 * DAYS_PER_YEAR,
        -0.00009515922545197159 * DAYS_PER_YEAR,
        0.0000515138902046611451 * SOLAR_MASS)

def advance(bodies, dt):
    n = len(bodies)
    i = 0
    while i < n:
        b = bodies[i]
        j = i + 1
        while j < n:
            b2 = bodies[j]
            dx = b[0] - b2[0]
            dy = b[1] - b2[1]
            dz = b[2] - b2[2]
            dist2 = dx * dx + dy * dy + dz * dz
            mag = dt / (dist2 * dist2 ** 0.5)
            b[3] = b[3] - dx * b2[6] * mag
            b[4] = b[4] - dy * b2[6] * mag
            b[5] = b[5] - dz * b2[6] * mag
            b2[3] = b2[3] + dx * b[6] * mag
            b2[4] = b2[4] + dy * b[6] * mag
            b2[5] = b2[5] + dz * b[6] * mag
            j = j + 1
        i = i + 1
    i = 0
    while i < n:
        b = bodies[i]
        b[0] = b[0] + dt * b[3]
        b[1] = b[1] + dt * b[4]
        b[2] = b[2] + dt * b[5]
        i = i + 1

def energy(bodies):
    e = 0.0
    n = len(bodies)
    i = 0
    while i < n:
        b = bodies[i]
        e = e + 0.5 * b[6] * (b[3]*b[3] + b[4]*b[4] + b[5]*b[5])
        j = i + 1
        while j < n:
            b2 = bodies[j]
            dx = b[0] - b2[0]
            dy = b[1] - b2[1]
            dz = b[2] - b2[2]
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5
            e = e - (b[6] * b2[6]) / dist
            j = j + 1
        i = i + 1
    return e

def offset_momentum(bodies):
    px = 0.0
    py = 0.0
    pz = 0.0
    i = 0
    while i < len(bodies):
        b = bodies[i]
        px = px + b[3] * b[6]
        py = py + b[4] * b[6]
        pz = pz + b[5] * b[6]
        i = i + 1
    sun = bodies[0]
    sun[3] = -px / SOLAR_MASS
    sun[4] = -py / SOLAR_MASS
    sun[5] = -pz / SOLAR_MASS

bodies = [make_sun(), make_jupiter(), make_saturn(), make_uranus(), make_neptune()]
offset_momentum(bodies)

n = 500000
i = 0
while i < n:
    advance(bodies, 0.01)
    i = i + 1
print(energy(bodies))
