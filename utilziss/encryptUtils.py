from lightphe import LightPHE
import numpy as np
from lightphe.models.Ciphertext import Ciphertext
from lightphe.models.Tensor import EncryptedTensor
from fractions import Fraction
from decimal import Decimal


class Encryptor:
    def __init__(
        self,
        partiallyHomomorphicAlgorithm: str = "Paillier",
        keyLength: int = 2048,
        keys: dict = None,
    ):
        self.partiallyHomomorphicAlgorithm = "Paillier"
        self.keyLength = 2048
        self.encryptorSystem = LightPHE(
            algorithm_name=partiallyHomomorphicAlgorithm, key_size=keyLength, keys=keys
        )

    def getKeys(self) -> dict:
        return self.encryptorSystem.cs.keys

    def setKeys(self, keys: dict = None, filePath: str = None) -> None:
        if not keys and not filePath:
            raise ValueError("No parameters given!")
        elif keys and filePath:
            raise ValueError("Expecting only one of the parameters")

        self.encryptorSystem = LightPHE(
            algorithm_name=self.partiallyHomomorphicAlgorithm,
            key_size=self.keyLength,
            key_file=filePath,
            keys=keys,
        )

    def savePublicKey(self, publicKeyFilePath: str) -> None:
        self.encryptorSystem.export_keys(public=True, target_file=publicKeyFilePath)

    def encrypt(
        self, plainValue: int | list[int] | list[float]
    ) -> int | list[int] | list[tuple[int, int]]:
        if isinstance(plainValue, int):
            encrypted = self.encryptorSystem.encrypt(plainValue)
            return encrypted.value
        elif isinstance(plainValue, list):
            if self.isAllInts(plainValue):
                return [
                    self.encryptorSystem.encrypt(value).value for value in plainValue
                ]
            elif self.isAllFloats(plainValue):
                plainFractionNumDenom = self.floatListToNumDenomTuples(plainValue)
                return [
                    (
                        self.encryptorSystem.encrypt(numerator).value,
                        self.encryptorSystem.encrypt(denominator).value,
                    )
                    for numerator, denominator in plainFractionNumDenom
                ]
            else:
                raise TypeError()
        else:
            raise TypeError()

    def floatListToNumDenomTuples(self, inputList):
        fractions = [Fraction(v) for v in inputList]
        fractionNumDenomTuples = [(f.numerator, f.denominator) for f in fractions]

        return fractionNumDenomTuples

    def isAllFloats(self, inputList):
        return all([isinstance(v, float) or isinstance(v, int) for v in inputList])

    def isAllInts(self, inputList):
        return all([isinstance(v, int) for v in inputList])

    def decrypt(self, cypherValue: int | list[int]) -> int | list[int]:
        if isinstance(cypherValue, int):
            cypherObject = self.encryptorSystem.create_ciphertext_obj(cypherValue)
            return self.encryptorSystem.create_ciphertext_obj(cypherObject)
        elif self.isAllInts(cypherValue):
            return [
                self.encryptorSystem.decrypt(
                    self.encryptorSystem.create_ciphertext_obj(value)
                )
                for value in cypherValue
            ]
        elif self.isAllFractions(cypherValue):
            decrypted = [
                (
                    self.encryptorSystem.decrypt(
                        self.encryptorSystem.create_ciphertext_obj(cnum)
                    ),
                    self.encryptorSystem.decrypt(
                        self.encryptorSystem.create_ciphertext_obj(cdenom)
                    ),
                )
                for cnum, cdenom in cypherValue
            ]
            npDecrypted = np.array(decrypted)
            convertToDecimal = np.vectorize(Decimal)
            nums = convertToDecimal(npDecrypted[:, 0])
            denoms = convertToDecimal(npDecrypted[:, 1])
            return [float(num / denom) for num,denom in zip(nums,denoms)]
        else:
            raise TypeError()

    def homomorphicAddition(
        self, cypherValue1: int | list[int], cypherValue2: int | list[int]
    ) -> int | list[int]:
        if isinstance(cypherValue1, int) and isinstance(cypherValue2, int):
            c1 = self.encryptorSystem.create_ciphertext_obj(cypherValue1)
            c2 = self.encryptorSystem.create_ciphertext_obj(cypherValue2)
            return (c1 + c2).value
        elif self.isAllInts(cypherValue1) and self.isAllInts(cypherValue2):
            return [
                (
                    self.encryptorSystem.create_ciphertext_obj(c1)
                    + self.encryptorSystem.create_ciphertext_obj(c2)
                ).value
                for c1, c2 in zip(cypherValue1, cypherValue2)
            ]
        elif (
            (self.isAllFloats(cypherValue1) and self.isAllFloats(cypherValue2))
            or (self.isAllFractions(cypherValue1) and self.isAllFractions(cypherValue2))
            or (self.isAllFloats(cypherValue1) and self.isAllFractions(cypherValue2))
            or (self.isAllFractions(cypherValue1) and self.isAllFloats(cypherValue2))
        ):
            cypherValue1Fractions = (
                cypherValue1
                if self.isAllFractions(cypherValue1)
                else self.floatListToNumDenomTuples(cypherValue1)
            )
            cypherValue2Fractions = (
                cypherValue2
                if self.isAllFractions(cypherValue2)
                else self.floatListToNumDenomTuples(cypherValue2)
            )
            print(
                "Warning: attempting to perform homomorphic addition on encrypted fractions. This mode only works properly when the second operand is plain fractions"
            )
            added = [
                (
                    (
                        self.encryptorSystem.create_ciphertext_obj(c1num) * c2denom
                        + self.encryptorSystem.create_ciphertext_obj(c1denom) * c2num
                    ).value,
                    (
                        self.encryptorSystem.create_ciphertext_obj(c1denom) * (c2denom)
                    ).value,
                )
                for (c1num, c1denom), (c2num, c2denom) in zip(
                    cypherValue1Fractions, cypherValue2Fractions
                )
            ]

            return added
        else:
            raise TypeError()

    def homomorphicMultiplication(
        self, cypherValue: int | list[int], plainValue: int | list[int]
    ) -> int | list[int]:
        if isinstance(cypherValue, int) and isinstance(plainValue, int):
            c1 = self.encryptorSystem.create_ciphertext_obj(cypherValue)
            return (c1 * plainValue).value
        elif isinstance(cypherValue, list) and isinstance(plainValue, list):
            isFractions = self.isAllFractions(cypherValue)
            if self.isAllInts(cypherValue) and self.isAllInts(plainValue):
                return [
                    (self.encryptorSystem.create_ciphertext_obj(c1) * multiplier).value
                    for c1, multiplier in zip(cypherValue, plainValue)
                ]
            elif isFractions or (
                self.isAllFloats(cypherValue) and self.isAllFloats(plainValue)
            ):
                cypherValueNumDenoms = (
                    cypherValue
                    if isFractions
                    else self.floatListToNumDenomTuples(cypherValue)
                )
                plainValueNumDenoms = (
                    plainValue
                    if self.isAllFractions(plainValue)
                    else self.floatListToNumDenomTuples(plainValue)
                )
                return [
                    (
                        (
                            self.encryptorSystem.create_ciphertext_obj(cnum)
                            * multiplierNum
                        ).value,
                        (
                            self.encryptorSystem.create_ciphertext_obj(cdenom)
                            * multiplierDenom
                        ).value,
                    )
                    for (cnum, cdenom), (multiplierNum, multiplierDenom) in zip(
                        cypherValueNumDenoms, plainValueNumDenoms
                    )
                ]
            else:
                raise TypeError()
        else:
            raise TypeError()

    def isAllFractions(self, inputList):
        return all(
            isinstance(v, tuple) and all(isinstance(vv, int) for vv in v)
            for v in inputList
        )


e = Encryptor()
A = [1, 2, 3.14]
EA = e.encrypt(A)
B = [1, 2.1, 4.1]
EB = e.encrypt(B)
EC = e.encrypt([4, 5, 6])

EApBm = e.homomorphicMultiplication(EA, B)
EApBa = e.homomorphicAddition(EA, B)
# EApBa = e.homomorphicAddition(EA, EC)
DApBm = e.decrypt(EApBm)
DApBa = e.decrypt(EApBa)
f = Fraction(3)


print()
