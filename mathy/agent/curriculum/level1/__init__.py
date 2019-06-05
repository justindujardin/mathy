from .white_belt import white_belt, white_belt_practice
from .yellow_belt import yellow_belt, yellow_belt_practice
from .green_belt import green_belt, green_belt_practice
from .purple_belt import purple_belt, purple_belt_practice
from .black_belt import black_belt, black_belt_practice
from .dev import dev, simple_polynomials

lessons = {
    "dev": dev,
    "poly3": simple_polynomials(3),
    "poly4": simple_polynomials(4),
    "poly5": simple_polynomials(5),
    "poly6": simple_polynomials(6),
    "poly7": simple_polynomials(7),
    "poly8": simple_polynomials(8),
    "poly9": simple_polynomials(9),
    "white_belt_practice": white_belt_practice,
    "white_belt": white_belt,
    "yellow_belt_practice": yellow_belt_practice,
    "yellow_belt": yellow_belt,
    "green_belt_practice": green_belt_practice,
    "green_belt": green_belt,
    "purple_belt_practice": purple_belt_practice,
    "purple_belt": purple_belt,
    "black_belt_practice": black_belt_practice,
    "black_belt": black_belt,
}

