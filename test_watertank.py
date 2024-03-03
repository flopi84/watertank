import unittest
from watertank import HotWaterTank
import numpy as np


class TestHotWaterTank(unittest.TestCase):
    def setUp(self):
        self.params = {
            "tank": {
                "height": 1.0,  # m
                "diameter": 0.5,  # m
                "thermal_transfer_coeff": 0.5,  # W/(m^2*K)
                "withdrawal_rate": 0.05,  # kg/s
                "min_regulation_temp": 64,  # °C
                "max_regulation_temp": 66,  # °C
            },
            "heater": {
                "efficiency": 0.9,
                "electrical_power": 10000,  # W
                "height": 0.2,  # m
            },
            "physics": {
                "water_thermal_conductivity": 0.6,  # W/(m*K)
                "water_cp": 4186,  # J/(kg*K)
                "water_rho": 1000,  # kg/m^3
                "convective_flow_speed": 0.05,  # m/s (estimate)
            },
            "simulation": {
                "num_disks": 5,
                "time": 12*3600,  # s
                "time_step": 5,  # s
            },
            "activation": {
                "diffusion": 1,
                "losses": 1,
                "convection": 1,
                "heater": 1,
                "withdrawal": 1,
            },
            "environment": {
                "ambient_temp": 20,  # Ambient Temperature
                "inflow_temp": 20  # Inflow Water Temperature
            },
            "scenario": {
                # Test scenario
                # - fill the tank with cold water
                # - (24h) plug in the tank and let it heat for about 24h (multiple temperature regulations) => loss estimations
                # - (12h) unplug the tank for withdrawals without heater (4 withdrawals per quarter of the tank)
                #   - perform a "Shower" withdrawal (8L/min for 5min) = about 15L/min of water at 40 so 8L/min of water at 65 for cold water at 18.
                #   - wait 55min to see the stratification stabilization
                #   - repeat a withdrawal 8L/min for 5min
                #   - etc
                # - (12h) plug the tank back in and let it reheat for about 12h
                # - perform withdrawals WITH heater (4 withdrawals per quarter of the tank)
                "plugged":   [[0,               1       ],
                            [24*3600,         0       ],
                            [36*3600,         1       ]
                            ],           
                "withdrawal": [[0,               0       ],
                            [24*3600,         8/60    ],
                            [24*3600+5*60,    0       ],
                            [25*3600,         8/60    ],
                            [25*3600+5*60,    0       ],
                            [26*3600,         8/60    ],
                            [26*3600+5*60,    0       ],
                            [27*3600,         8/60    ],
                            [27*3600+5*60,    0       ],
                            [36*3600,         8/60    ],
                            [36*3600+5*60,    0       ],
                            [37*3600,         8/60    ],
                            [37*3600+5*60,    0       ],
                            [38*3600,         8/60    ],
                            [38*3600+5*60,    0       ],
                            [39*3600,         8/60    ],
                            [39*3600+5*60,    0       ]
                            ]      
            
            }
        }
        self.tank = HotWaterTank(self.params)

    def test_initialize_variables(self):
        """Test if variables are initialized correctly."""
        expected_disk_height = 0.2  # m
        expected_disk_surface = 0.19634954084
        expected_water_mass_per_disk = 39.2699081699
        expected_total_water_volume = 196.349540849 # in Liters
        expected_lateral_surface_per_disk = 0.31415926535
        expected_thermal_diffusion_coeff = 0.00000358337
        
        self.assertAlmostEqual(self.tank.variables["disk_height"], expected_disk_height)
        self.assertAlmostEqual(self.tank.variables["disk_surface"], expected_disk_surface)
        self.assertAlmostEqual(self.tank.variables["water_mass_per_disk"], expected_water_mass_per_disk)
        self.assertAlmostEqual(self.tank.variables["total_water_volume"], expected_total_water_volume)
        self.assertAlmostEqual(self.tank.variables["lateral_surface_per_disk"], expected_lateral_surface_per_disk)
        self.assertAlmostEqual(self.tank.variables["thermal_diffusion_coeff"], expected_thermal_diffusion_coeff)
        # Initial temperature vector test
        initial_temp_vector = np.ones(7) * 20
        np.testing.assert_array_almost_equal(self.tank.variables["temperature_vector"], initial_temp_vector)


    def test_calculate_matrices(self):
        """Test if matrices are calculated correctly."""
        self.tank.calculate_matrices()
        # Test the shapes of matrices
        self.assertEqual(self.tank.Adiff.shape, (7, 7))
        self.assertEqual(self.tank.Adraw.shape, (7, 7))
        self.assertEqual(self.tank.Aloss.shape, (7, 7))
        self.assertEqual(self.tank.Acircflow.shape, (7, 7))
        self.assertEqual(self.tank.B.shape, (7,))
        # Test matrix values
        expected_B = [0.        , 0.05474991, 0.        , 0.        , 0.        ,       0.        , 0.        ]
        expected_Adiff = [
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00],
        [ 0.00000000e+00, -3.58337315e-06,  3.58337315e-06,         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00],
        [ 0.00000000e+00,  3.58337315e-06, -7.16674630e-06,         3.58337315e-06,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  3.58337315e-06,        -7.16674630e-06,  3.58337315e-06,  0.00000000e+00,         0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         3.58337315e-06, -7.16674630e-06,  3.58337315e-06,         0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00,  3.58337315e-06, -3.58337315e-06,         0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00]]
        expected_Adraw = [
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,         0.        ,  0.        ],
        [ 0.02546479, -0.02546479,  0.        ,  0.        ,  0.        ,         0.        ,  0.        ],
        [ 0.        ,  0.02546479, -0.02546479,  0.        ,  0.        ,         0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.02546479, -0.02546479,  0.        ,         0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.02546479, -0.02546479,         0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.02546479,        -0.02546479,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,         0.        ,  0.        ]]
        expected_Aloss = [
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00],
        [ 0.00000000e+00, -1.55279503e-06,  0.00000000e+00,         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         1.55279503e-06],
        [ 0.00000000e+00,  0.00000000e+00, -9.55566173e-07,         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         9.55566173e-07],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,        -9.55566173e-07,  0.00000000e+00,  0.00000000e+00,         9.55566173e-07],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00, -9.55566173e-07,  0.00000000e+00,         9.55566173e-07],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00,  0.00000000e+00, -1.55279503e-06,         1.55279503e-06],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,         0.00000000e+00]]
        expected_Acircflow = [
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  , -0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.25, -0.5 ,  0.25,  0.  ,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.25, -0.5 ,  0.25,  0.  ,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.25, -0.5 ,  0.25,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.25, -0.25,  0.  ],
        [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]]
        
        np.testing.assert_array_almost_equal(self.tank.B, expected_B)
        np.testing.assert_array_almost_equal(self.tank.Adiff, expected_Adiff)
        np.testing.assert_array_almost_equal(self.tank.Adraw, expected_Adraw)
        np.testing.assert_array_almost_equal(self.tank.Aloss, expected_Aloss)
        np.testing.assert_array_almost_equal(self.tank.Acircflow, expected_Acircflow)
        
    def test_simulation_results(self):
        """Test the simulation results for consistency."""
        self.tank.simulate()
        # Test if results are stored correctly
        self.assertTrue(isinstance(self.tank.results, np.ndarray))
        # Add more detailed tests

if __name__ == '__main__':
    unittest.main()
