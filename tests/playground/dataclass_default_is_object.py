from dataclasses import field, dataclass


class A:
    n: float = 0.2

@dataclass
class B:
    a_instance: A = A()

@dataclass
class C:
    a_instance: A = field(default_factory=A)

x = B()

x.a_instance.n = 1.2

y = B()
print(y.a_instance.n)


cx = C()
cx.a_instance.n=2.3
cy = C()
print(cy.a_instance.n)

