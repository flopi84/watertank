import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

class HotWaterTank:
    def __init__(self, params):
        self.params = params
        self.variables = {}  # Initializing self.variables
        self.initialize_variables()
        self.calculate_matrices()

    def initialize_variables(self):
        """Initializes the necessary variables for the simulation from the hot water tank parameters."""
        # No need to pass params as an argument since we access self.params
        
        # Calculations based on tank parameters
        disk_surface = np.pi * (self.params["tank"]["diameter"] / 2)**2
        disk_height = self.params["tank"]["height"] / self.params["simulation"]["num_disks"]
        water_mass_per_disk = disk_surface * disk_height * self.params["physics"]["water_rho"]
        total_water_volume = disk_surface * self.params["tank"]["height"] * 1000  # in Liters

        lateral_surface_per_disk = np.pi * self.params["tank"]["diameter"] * disk_height

        # Calculations based on physical parameters
        thermal_diffusion_coeff = self.params["physics"]["water_thermal_conductivity"] / (self.params["physics"]["water_rho"] * self.params["physics"]["water_cp"] * disk_height**2)

        # Recording the calculated variables in self.variables
        self.variables["disk_surface"] = disk_surface
        self.variables["disk_height"] = disk_height
        self.variables["water_mass_per_disk"] = water_mass_per_disk
        self.variables["total_water_volume"] = total_water_volume
        self.variables["lateral_surface_per_disk"] = lateral_surface_per_disk
        self.variables["thermal_diffusion_coeff"] = thermal_diffusion_coeff

        # Initial temperature vector
        temperature_vector = np.ones(self.params["simulation"]["num_disks"]+2) * self.params["environment"]["inflow_temp"]
        temperature_vector[0] = self.params["environment"]["inflow_temp"]
        temperature_vector[-1] = self.params["environment"]["ambient_temp"]

        self.variables["temperature_vector"] = temperature_vector

        # Time step limitation for convergence
        withdrawal_speed = self.params["tank"]["withdrawal_rate"] / self.params["physics"]["water_rho"] / disk_surface
        time_step = min(self.params["simulation"]["time_step"], disk_height 
                        / (self.params["physics"]["convective_flow_speed"]+withdrawal_speed)/ 2) # s (max disk height / convective flow speed / 2)

        num_steps = int(self.params["simulation"]["time"] / time_step) + 1  # Total number of time steps

        self.variables["time_step"] = time_step
        self.variables["num_steps"] = num_steps

    def calculate_matrices(self):
        """Calculates the matrices used for the simulation using self.variables."""
        num_disks = self.params["simulation"]["num_disks"]
        
        # Using pre-calculated variables
        disk_height = self.variables["disk_height"]
        disk_surface = self.variables["disk_surface"]
        water_mass_per_disk = self.variables["water_mass_per_disk"]
        lateral_surface_per_disk = self.variables["lateral_surface_per_disk"]
        thermal_diffusion_coeff = self.variables["thermal_diffusion_coeff"]
        
        # Initializing matrices
        self.Adiff = np.zeros((num_disks + 2, num_disks + 2))  # Diffusion
        self.Adraw = np.zeros((num_disks + 2, num_disks + 2))  # Withdrawal
        self.Aloss = np.zeros((num_disks + 2, num_disks + 2))  # Losses
        self.Acircflow = np.zeros((num_disks + 2, num_disks + 2))  # Convection
        self.B = np.zeros(num_disks + 2)  # Heat input from the heater
        


        # Adiff = diffusion matrix = thermal diffusion between disks  
        # disk 1 (bottom)
        self.Adiff[1][1] = -thermal_diffusion_coeff
        self.Adiff[1][2] = thermal_diffusion_coeff
        # disk N (top)
        self.Adiff[num_disks][num_disks] = -thermal_diffusion_coeff
        self.Adiff[num_disks][num_disks-1] = thermal_diffusion_coeff

        for i in range(2, num_disks):
            self.Adiff[i][i] = -2 * thermal_diffusion_coeff
            self.Adiff[i][i - 1] = thermal_diffusion_coeff
            self.Adiff[i][i + 1] = thermal_diffusion_coeff
        self.Adiff *= self.params["activation"]["diffusion"]
        
        # Adraw = Withdrawal
        for i in range(1, num_disks + 1):
            self.Adraw[i][i-1] = 1/water_mass_per_disk
            self.Adraw[i][i] = -1/water_mass_per_disk
        self.Adraw *= self.params["activation"]["withdrawal"]
        
        # Aloss - Losses
        self.Aloss[1][1] = -(disk_surface + lateral_surface_per_disk) * self.params["tank"]["thermal_transfer_coeff"] / (self.params["physics"]["water_cp"] * water_mass_per_disk)
        self.Aloss[1][num_disks+1] = -self.Aloss[1][1]
        self.Aloss[num_disks][num_disks] = self.Aloss[1][1]
        self.Aloss[num_disks][num_disks+1] = -self.Aloss[1][1]
        for i in range(2, num_disks):
            self.Aloss[i][i] = -lateral_surface_per_disk * self.params["tank"]["thermal_transfer_coeff"] / (self.params["physics"]["water_cp"] * water_mass_per_disk)
            self.Aloss[i][num_disks+1] = -self.Aloss[i][i]
        self.Aloss *= self.params["activation"]["losses"]

        # Acircflow - Convection
        convective_flow_rate = self.params["physics"]["convective_flow_speed"] / disk_height * water_mass_per_disk
        convection_term = convective_flow_rate / water_mass_per_disk
        self.Acircflow[1][1] = -convection_term
        self.Acircflow[1][2] = convection_term
        self.Acircflow[num_disks][num_disks] = -convection_term
        self.Acircflow[num_disks][num_disks-1] = convection_term

        for i in range(2, num_disks):
            self.Acircflow[i][i] = -2*convection_term
            self.Acircflow[i][i-1] = convection_term
            self.Acircflow[i][i+1] = convection_term
        self.Acircflow *= self.params["activation"]["convection"]

        # B - Heat input from the heater
        # ----- array of heat input coefficients per disk
        # ----- it's the coeff of power input in each disk, depending on the height 
        # ----- of the heater, the tank, and the number of disks
        Heat_input_coeff = np.zeros(num_disks)  # array of 0
        num_heated_disks = self.params["heater"]["height"] / disk_height #not an integer number
        for i in range(0, num_disks-1):
            if (i+1) <= math.floor(num_heated_disks):
                Heat_input_coeff[i] = 1 / num_heated_disks
            elif (i+1) == math.ceil(num_heated_disks):
                Heat_input_coeff[i] = (num_heated_disks - math.floor(num_heated_disks)) / num_heated_disks
            else:
                Heat_input_coeff[i] = 0

        constant = self.params["heater"]["electrical_power"] * self.params["heater"]["efficiency"] / (water_mass_per_disk * self.params["physics"]["water_cp"])
        self.B = np.concatenate(([0], constant * Heat_input_coeff, [0]))
        self.B *= self.params["activation"]["heater"]

    def simulate(self):
        #Performs the thermal simulation of the hot water tank and stores the results in self.results
        temperature_vector = self.variables["temperature_vector"]
        num_disks = self.params["simulation"]["num_disks"]
        simulation_time = self.params["simulation"]["time"]
        time_step = self.variables["time_step"]
        max_regul_temp = self.params["tank"]["max_regulation_temp"]
        min_regul_temp = self.params["tank"]["min_regulation_temp"]
        num_steps = self.variables["num_steps"]
        withdrawal = np.array(self.params["scenario"]["withdrawal"])
        plugged = np.array(self.params["scenario"]["plugged"])

        results = np.zeros((num_disks, num_steps))
        results[:,0] = temperature_vector[1:num_disks+1]

        # Scenario vectorization
        self.time_vector = np.arange(0, simulation_time, time_step)
        withdrawal_indices = np.maximum(0, np.searchsorted(withdrawal[:,0], self.time_vector, side='left') - 1)
        plug_indices = np.maximum(0, np.searchsorted(plugged[:,0], self.time_vector, side='left') - 1)
        self.withdrawal_vectorized = withdrawal[withdrawal_indices,1]
        self.plugged_vectorized = plugged[plug_indices,1]

        # Simulation loop
        u = 0
        for i in range(1, num_steps):

            # Scenario
            time = i * time_step
            withdrawal_rate = self.withdrawal_vectorized[i]
            plugged = self.plugged_vectorized[i]

            # Temperature regulation
            u = np.where(plugged == 1, np.where(temperature_vector[1] > max_regul_temp, 0, np.where(temperature_vector[1] < min_regul_temp, 1, u)), 0)

            # Update temperature vector
            A = self.Adiff + np.dot(self.Adraw, withdrawal_rate) + np.dot(self.Acircflow, u) + self.Aloss
            temperature_change_vector = np.dot(A, temperature_vector) + np.dot(self.B, u)
            temperature_vector += temperature_change_vector * time_step
            
            # Record temperatures for each disk
            results[:, i] = temperature_vector[1:num_disks+1]  # Exclude Te and Tamb
            
            print(time)
        # Store results for display
        self.results = results

    def display_results(self):
        heights = np.arange(self.variables["disk_height"]/2, self.params["tank"]["height"], self.variables["disk_height"])

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True, 
                                gridspec_kw={'height_ratios': [3, 1]})

        # Chart 1: Temperature distribution
        im = axs[0].imshow(self.results, aspect='auto', origin='lower',
                        extent=[self.time_vector.min(), self.time_vector.max(), 0, self.params["tank"]["height"]],
                        vmin=15, vmax=65, cmap='coolwarm', interpolation='nearest')
        fig.colorbar(im, ax=axs[0], label='Temperature (°C)')
        for breakpoint in heights:
            axs[0].axhline(y=breakpoint+self.variables["disk_height"]/2, color='r', linestyle='--', linewidth=0.5)
        axs[0].set_ylabel('Height')
        axs[0].set_title(f'Tank Temperature - Time step = {self.variables["time_step"]:.4f} seconds')

        # Chart 2: 0 or 1 values over time
        axs[1].plot(self.time_vector, self.plugged_vectorized, label='Plugged in or not')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Value')
        axs[1].legend()

        # Second y-axis for the same x-axis
        ax2 = axs[1].twinx()

        ax2.plot(self.time_vector, self.withdrawal_vectorized*60, label='withdrawal rate', color='green', linestyle='--')
        ax2.set_ylabel('rate L/min', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.legend(loc='upper right')

        # New figure and axes for temperature curves
        plt.figure()
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.title('Temperature over time for different heights')

        # Plot temperature curves
        for i in range(self.params["simulation"]["num_disks"]):
            plt.plot(self.time_vector, self.results[i], label=f'Height {i}')

        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.title('Temperature over time for different heights')
        plt.legend()
        plt.grid(True)
        plt.ylim(15, 70)
        plt.gca().xaxis.set_major_formatter(time_formatter)
        plt.show()

def load_parameters():
    """Loads the simulation parameters for the hot water tank, organized by categories."""
    params = {
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
            "electrical_power": 2400,  # W
            "height": 0.35,  # m
        },
        "physics": {
            "water_thermal_conductivity": 0.6,  # W/(m*K)
            "water_cp": 4186,  # J/(kg*K)
            "water_rho": 1000,  # kg/m^3
            "convective_flow_speed": 0.05,  # m/s (estimate)
        },
        "simulation": {
            "num_disks": 5,
            "time": 48*3600,  # s
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
    return params

# Function to convert seconds to hh:mm:ss
def format_time(x, pos):
    hours = int(x // 3600)
    minutes = int((x % 3600) // 60)
    seconds = int(x % 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
time_formatter = FuncFormatter(format_time)

def main():
    params = load_parameters()
    tank = HotWaterTank(params)
    tank.simulate()
    tank.display_results()

if __name__ == "__main__":
    main()
