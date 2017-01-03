from pyspark.mllib.common import JavaModelWrapper
from pyspark.mllib.linalg.distributed import RowMatrix

class SVD(JavaModelWrapper):
    @property
    def U(self):
        u = self.call("U")
        if u is not None:
            return RowMatrix(u)
    @property
    def s(self):
        return self.call("s")
    @property
    def V(self):
        return self.call("V")

def computeSVD(row_matrix, k, computeU=False, rCond=1e-9):
    java_model = row_matrix._java_matrix_wrapper.call("computeSVD", int(k), computeU, float(rCond))
    return SVD(java_model)