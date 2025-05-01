
import taichi as ti
import numpy as np
import time

@ti.kernel
def xpay(z: ti.template(), a: float, x: ti.template(), y: ti.template(), A: ti.template()):
    for i in z:
        z[i] = x[i] + a * y[i]

@ti.kernel
def dot(x: ti.template(), y: ti.template(), A: ti.template()) -> ti.f32:
    result = 0.0
    # I believe Taichi optimizes this
    for i in x:
        result += x[i] * y[i]
    return result

@ti.dataclass
class Offdiag:
    e1: ti.int32
    e2: ti.int32
    e3: ti.int32
    e4: ti.int32
    e5: ti.int32
    e6: ti.int32

@ti.data_oriented
class CG:
    def __init__(self, n):
        self.b = ti.field(dtype=ti.f32, shape=n)
        self.d = ti.field(dtype=ti.f32, shape=n)
        self.x = ti.field(dtype=ti.f32, shape=n)
        self.q = ti.field(dtype=ti.f32, shape=n)
        self.r = ti.field(dtype=ti.f32, shape=n)

        self.Adiag = ti.field(dtype=ti.i32, shape=n)
        self.Aoff = Offdiag.field(shape=(n))

        self.matmultime = 0.0
        self.vectortime = 0.0
        self.reducetime = 0.0
        self.cgiters = 0

    @ti.func
    def getd(self, i):
        ret = 0.0
        if i >= 0:
            ret = self.d[i]
        return ret

    @ti.kernel
    def Adq(self):
        for i in self.q:
            if self.Adiag[i] > 0:
                qi = self.Adiag[i] * self.d[i]
                od = self.Aoff[i]
                qi -= self.getd(od.e1)
                qi -= self.getd(od.e2)
                qi -= self.getd(od.e3)
                qi -= self.getd(od.e4)
                qi -= self.getd(od.e5)
                qi -= self.getd(od.e6)
                self.q[i] = qi
            else:
                self.q[i] = 0.0

    def solve(self):
        steps = 0
        self.x.fill(0.0)
        self.r.copy_from(self.b)
        self.d.copy_from(self.b)
        sigma = dot(self.r, self.r, self.Adiag)
        tol = sigma * (1e-6)
        while sigma > tol and steps < 100:
            steps += 1

            start_time = time.time()
            self.Adq()
            self.matmultime += time.time() - start_time

            start_time = time.time()
            alpha = sigma / dot(self.d, self.q, self.Adiag)
            self.reducetime += time.time() - start_time

            start_time = time.time()
            xpay(self.x, alpha, self.x, self.d, self.Adiag)
            xpay(self.r, -alpha, self.r, self.q, self.Adiag)
            sigmaold = sigma
            self.vectortime += time.time() - start_time

            start_time = time.time()
            sigma = dot(self.r, self.r, self.Adiag)
            self.reducetime += time.time() - start_time

            start_time = time.time()
            beta = sigma / sigmaold
            xpay(self.d, beta, self.r, self.d, self.Adiag)
            self.vectortime += time.time() - start_time
        #print("%s steps, %s residual" % (steps, sigma))
                

if __name__ == "__main__":

    ti.init(arch=ti.cpu)
    solver = CG(3)

    empty = np.array([-1, -1, -1], dtype=int)
    solver.A1.from_numpy(empty)
    solver.A2.from_numpy(empty)
    solver.A3.from_numpy(empty)
    solver.A4.from_numpy(empty)
    empty = np.array([1, 0, 1], dtype=int)
    solver.A5.from_numpy(empty)
    empty = np.array([-1, 2, -1], dtype=int)
    solver.A6.from_numpy(empty)
    empty = np.array([1, 2, 1], dtype=int)
    solver.Adiag.from_numpy(empty)

    b = np.array([2., 3., -5.])
    solver.b.from_numpy(b)

    solver.solve()
    print(solver.x.to_numpy())
            


