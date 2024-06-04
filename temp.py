import numpy as np

# Define the fuel map and index map
fuel_map_episode = [
    1,
    2,
    3,
    np.random.choice([1, 2, 3]),
    4,
    5,
    6,
    np.random.choice([4, 5, 6]),
    np.random.randint(1, 7),
    8,
    9,
    10,
    np.random.choice([8, 9, 10]),
    np.random.randint(4, 11),
    np.random.randint(1, 11),
]

index_map = np.array(
    [
        600,
        1200,
        1800,
        2500,
        3100,
        3700,
        4500,
        5100,
        5700,
        6300,
        7000,
        7600,
        8500,
        9100,
    ]
)

# One-liner to get the fuel value
global_step = 6299  # Example episode number
fuel_value = fuel_map_episode[np.sum(global_step >= index_map)]

print(f"Episode {global_step}: Fuel Value = {fuel_value}")
