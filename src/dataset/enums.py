from __future__ import annotations
import math
from enum import Flag


class VisitType(Flag):
    """
    Visit Type represented as flag enum.
    Some preanesthesia powernotes check multiple visit types.  This enum
    can represent all possible combinations and provides methods to reduce
    the visit type down to a single visit type based on priority.
    """

    NONE = 0
    IN_PERSON = 1
    PHONE_EVALUATION = 2
    CHART_REVIEW = 4

    @classmethod
    def from_string(cls, text: str) -> VisitType:
        "Converts visit type text to Enum"
        flags = VisitType(0)
        if not bool(text):
            return flags
        else:
            text = text.lower()
            if "in person" in text:
                flags = flags | cls.IN_PERSON
            if "phone evaluation" in text:
                flags = flags | cls.PHONE_EVALUATION
            if "chart review" in text:
                flags = flags | cls.CHART_REVIEW
            return flags

    def count_flags(self) -> int:
        "Count number of flags set to True."
        return bin(self._value_).count("1")

    def reduce(self) -> VisitType:
        "If multiple visit type flags, reduce to single visit type flag by priority."
        if self.count_flags() > 1:
            if VisitType.IN_PERSON in self:
                return VisitType.IN_PERSON
            elif VisitType.PHONE_EVALUATION in self:
                return VisitType.PHONE_EVALUATION
            elif VisitType.CHART_REVIEW in self:
                return VisitType.CHART_REVIEW
            else:
                raise Exception("Unknown VisitType Flag")
        else:
            return self

    def serialize(self) -> str:
        "Convert enum object to string representation for saving into file."
        # Dict mapping enum integer to names
        int2name = {
            k: v.name
            for k, v in VisitType._value2member_map_.items()
            if v.name is not None
        }
        # Convert integers to LSB position in bit_flags
        bin2name = {int(math.log(k, 2)): v for k, v in int2name.items() if k != 0}
        # Get Binary representatioin of decimal value of Enum
        bit_flags = [int(x) for x in bin(self._value_)[2:]]
        # Reverse bit_flags so LSB becomes MSB to fit python indexing convention
        bit_flags.reverse()
        # Get Names for positions set to one
        flag_names = [bin2name[index] for index, x in enumerate(bit_flags) if x == 1]
        flag_names_str = " ".join(flag_names)
        return flag_names_str

    @classmethod
    def deserialize(cls, flag_names_str: str) -> VisitType:
        "Convert serialized string representation to enum object."
        # Dict mapping enum names to integers
        name2int = {
            v.name: k for k, v in cls._value2member_map_.items() if v.name is not None
        }
        flag_names = [
            name if (name != "") else "NONE" for name in flag_names_str.split(" ")
        ]
        decimal_value = sum([name2int[flag_name] for flag_name in flag_names])
        return cls(decimal_value)


class RCRI(Flag):
    """
    Revised Cardiac Risk Index represented as flag enum.
    Enum can represent any combination of categorical values
    within the RCRI Enum, and can use `score()` to count the
    total number of items in enum, which is equivalent to RCRI score.
    Can also transform the enum into a set and reconstruct enum
    object from a set.
    """

    NONE = 0
    HIGH_RISK_SURGERY = 1
    HISTORY_ISCHEMIC_HEART_DISEASE = 2
    HISTORY_CONGESTIVE_HEART_FAILURE = 4
    HISTORY_CEREBROVASCULAR_DISEASE = 8
    PREOPERATIVE_INSULIN = 16
    PREOPERATIVE_CREATININE = 32

    @classmethod
    def from_string(cls, text: str) -> RCRI:
        "Converts structured powernote text into Enum."
        flags = RCRI(0)
        if not bool(text):
            return flags
        else:
            text = text.lower()
            if "none." in text:
                return flags
            if "high risk surgery" in text:
                flags = flags | cls.HIGH_RISK_SURGERY
            if "ischemic heart" in text:
                flags = flags | cls.HISTORY_ISCHEMIC_HEART_DISEASE
            if "congestive heart" in text:
                flags = flags | cls.HISTORY_CONGESTIVE_HEART_FAILURE
            if "cerebrovascular disease" in text:
                flags = flags | cls.HISTORY_CEREBROVASCULAR_DISEASE
            if "insulin" in text:
                flags = flags | cls.PREOPERATIVE_INSULIN
            if "creatinine" in text:
                flags = flags | cls.PREOPERATIVE_CREATININE
            return flags

    def count_flags(self) -> int:
        "Count number of flags set to True."
        return bin(self._value_).count("1")

    def serialize(self) -> str:
        "Convert enum object to string representation for saving into file."
        # Dict mapping enum integer to names
        int2name = {
            k: v.name for k, v in RCRI._value2member_map_.items() if v.name is not None
        }
        # Convert integers to LSB position in bit_flags
        bin2name = {int(math.log(k, 2)): v for k, v in int2name.items() if k != 0}
        # Get Binary representatioin of decimal value of Enum
        bit_flags = [int(x) for x in bin(self._value_)[2:]]
        # Reverse bit_flags so LSB becomes MSB to fit python indexing convention
        bit_flags.reverse()
        # Get Names for positions set to one
        flag_names = [bin2name[index] for index, x in enumerate(bit_flags) if x == 1]
        flag_names_str = " ".join(flag_names)
        return flag_names_str

    @classmethod
    def deserialize(cls, flag_names_str: str) -> RCRI:
        "Convert serialized string representation to enum object."
        # Dict mapping enum names to integers
        name2int = {
            v.name: k for k, v in cls._value2member_map_.items() if v.name is not None
        }
        flag_names = [
            name if (name != "") else "NONE" for name in flag_names_str.split(" ")
        ]
        decimal_value = sum([name2int[flag_name] for flag_name in flag_names])
        return cls(decimal_value)