# global
global_variable = 1

# function
def function():
    local_variable = 2

# closure
def outer():
    def inner():
        pass

# nested closure
def level_one():
    def level_two():
        def level_three():
            pass

# class with member and methods
class ClassDef:

    # class member
    class_member = 0

    # method
    def method(self):
        local_variable = 3

    # classmethod
    @classmethod
    def class_method():
        pass

    # staticmethod
    @staticmethod
    def static_method():
        pass
