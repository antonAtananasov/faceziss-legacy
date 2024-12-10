import tenseal as ts
from tenseal import CKKSVector
import numpy as np


# Setup TenSEAL context
class Encryptor:
    def __init__(self):
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40

    def encryptArray(self, inputArray):
        return ts.ckks_vector(self.context, inputArray)

    def decryptArray(self, cipherVector):
        return cipherVector.decrypt()

    def add(self, cipherVector1, cipherVector2):
        return cipherVector1 + cipherVector2

    def dot(self, cipherVector1, cipherVector2):
        return cipherVector1.dot(cipherVector2)

    def matrixMultiplication(self, cipherVector, plainMatrix):
        return cipherVector.matmul(plainMatrix)


if __name__ == "__main__":
    # # Setup TenSEAL context
    # context = ts.context(
    #     ts.SCHEME_TYPE.CKKS,
    #     poly_modulus_degree=8192,
    #     coeff_mod_bit_sizes=[60, 40, 40, 60],
    # )
    # context.generate_galois_keys()
    # context.global_scale = 2**40

    # v1 = [0, 1, 2, 3, 4]
    # v2 = [4, 3, 2, 1, 0]

    # # encrypted vectors
    # enc_v1 = ts.ckks_vector(context, v1)
    # enc_v2 = ts.ckks_vector(context, v2)

    # result = enc_v1 + enc_v2
    # result.decrypt()  # ~ [4, 4, 4, 4, 4]

    # result = enc_v1.dot(enc_v2)
    # result.decrypt()  # ~ [10]

    # matrix = [
    #     [73, 0.5, 8],
    #     [81, -5, 66],
    #     [-100, -78, -2],
    #     [0, 9, 17],
    #     [69, 11, 10],
    # ]
    # result = enc_v1.matmul(matrix)
    # result.decrypt()  # ~ [157, -90, 153]

    e = Encryptor()

    A = [1, -2, np.pi, -np.pi]
    EA = e.encryptArray(A)
    B = [-3, -1, 2, np.e]
    EB = e.encryptArray(B)

    EaEb = e.add(EA, EB)
    Mab = e.decryptArray(EaEb)

    print()
