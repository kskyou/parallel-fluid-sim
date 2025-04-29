
import taichi as ti
import numpy as np
import time


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

        self.n = n
        self.numalive = n
        self.indexpoint = ti.field(dtype=ti.i32, shape=n)
        self.prefix_sum = ti.field(dtype=ti.i32, shape=n)
        self.indexin = ti.field(dtype=ti.i32, shape=n)

        self.matmultime = 0.0
        self.vectortime = 0.0
        self.cgiters = 0

    @ti.func
    def getd(self, i):
        ret = 0.0
        if i >= 0:
            ret = self.d[i]
        return ret

    @ti.kernel
    def Adq(self, numalive: int):
        for j in range(numalive):
            i = self.indexin[j]
            qi = self.Adiag[i] * self.d[i]
            od = self.Aoff[i]
            qi -= self.getd(od.e1)
            qi -= self.getd(od.e2)
            qi -= self.getd(od.e3)
            qi -= self.getd(od.e4)
            qi -= self.getd(od.e5)
            qi -= self.getd(od.e6)
            self.q[i] = qi

    @ti.kernel
    def xpay(self, z: ti.template(), a: float, x: ti.template(), y: ti.template(), numalive: int):
        for j in range(numalive):
            i = self.indexin[j]
            z[i] = x[i] + a * y[i]

    @ti.kernel
    def dot(self, x: ti.template(), y: ti.template(), numalive: int) -> ti.f32:
        result = 0.0
        for j in range(numalive):
            i = self.indexin[j]
            result += x[i] * y[i]
        return result

    @ti.kernel
    def identify(self) -> ti.i32:
        for i in self.Adiag:
            if self.Adiag[i] > 0:
                self.indexpoint[i] = 1
            else:
                self.indexpoint[i] = 0

        self.prefix_sum[0] = 0
        ti.loop_config(serialize=True)
        for i in range(1, self.n):
            self.prefix_sum[i] = self.prefix_sum[i - 1] + self.indexpoint[i-1]

        numalive = self.prefix_sum[self.n - 1] + self.indexpoint[self.n - 1]
        for i in range(0, self.n - 1):
            if self.prefix_sum[i+1] > self.prefix_sum[i]:
                self.indexin[self.prefix_sum[i]] = i
        if self.indexpoint[self.n - 1] == 1:
            self.indexin[self.prefix_sum[self.n - 1]] = self.n - 1
        return numalive

    def makeindex(self):
        self.prefix_sum.fill(0)
        self.indexin.fill(0)
        self.numalive = self.identify()

    def solve(self):
        self.makeindex()
        steps = 0
        self.x.fill(0.0)
        self.q.fill(0.0)
        self.r.copy_from(self.b)
        self.d.copy_from(self.b)
        sigma = self.dot(self.r, self.r, self.numalive)
        tol = sigma * (1e-6)
    
        while sigma > tol and steps < 100:
            steps += 1

            start_time = time.time()
            self.Adq(self.numalive)
            self.matmultime += time.time() - start_time

            start_time = time.time()
            alpha = sigma / self.dot(self.d, self.q, self.numalive)
            self.xpay(self.x, alpha, self.x, self.d, self.numalive)
            self.xpay(self.r, -alpha, self.r, self.q, self.numalive)
            sigmaold = sigma
            sigma = self.dot(self.r, self.r, self.numalive)
            beta = sigma / sigmaold
            self.xpay(self.d, beta, self.r, self.d, self.numalive)

            self.vectortime += time.time() - start_time
        #print("%s steps, %s residual" % (steps, sigma))
        self.cgiters += steps
                

if __name__ == "__main__":

    ti.init(arch=ti.cpu)
    solver = CG(8)

    empty = np.array([0, 1, 0, 1, 1, 1, 1, 1], dtype=int)
    solver.Adiag.from_numpy(empty)

    solver.makeindex()

    print(solver.indexin.to_numpy())
    print(solver.numalive)
            


