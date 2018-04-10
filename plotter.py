import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plotter():
    def __init__(self):
        self.labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4', 'reward']
        self.cache = {x : [] for x in self.labels}
        
    def add(self, data_to_log):
        for ii in range(len(self.labels)):
            self.cache[self.labels[ii]].append(data_to_log[ii])
            
    def plot_reward(self):
        plt.plot(self.cache['time'], self.cache['reward'], label='reward')
        plt.legend()
        _ = plt.ylim()
        
    def plot_trajectory(self, target_pose=None):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(self.cache['x'], self.cache['y'], self.cache['z'], label='trajectory')
        
        ax.scatter(self.cache['x'][0], self.cache['y'][0], self.cache['z'][0], c='g', marker='o', s=20, label='start')
        ax.scatter(self.cache['x'][-1], self.cache['y'][-1], self.cache['z'][-1], c='r', marker='o', s=20, label='end')
        if target_pose is not None:
            ax.scatter(target_pose[0], target_pose[1], target_pose[2], c='y', marker='o', s=20, label='target')
            
        ax.legend()
        plt.show()
        
    def plot_position2d(self):
        plt.plot(self.cache['time'], self.cache['x'], label='x')
        plt.plot(self.cache['time'], self.cache['y'], label='y')
        plt.plot(self.cache['time'], self.cache['z'], label='z')
        plt.legend()
        _ = plt.ylim()
        
    def plot_velocity(self):
        plt.plot(self.cache['time'], self.cache['x_velocity'], label='x_hat')
        plt.plot(self.cache['time'], self.cache['y_velocity'], label='y_hat')
        plt.plot(self.cache['time'], self.cache['z_velocity'], label='z_hat')
        plt.legend()
        _ = plt.ylim()
        
    def plot_euler_agnels(self):
        plt.plot(self.cache['time'], self.cache['phi'], label='phi')
        plt.plot(self.cache['time'], self.cache['theta'], label='theta')
        plt.plot(self.cache['time'], self.cache['psi'], label='psi')
        plt.legend()
        _ = plt.ylim()
        
    def plot_rotor_revs(self):
        plt.plot(self.cache['time'], self.cache['rotor_speed1'], label='r_1')
        plt.plot(self.cache['time'], self.cache['rotor_speed2'], label='r_2')
        plt.plot(self.cache['time'], self.cache['rotor_speed3'], label='r_3')
        plt.plot(self.cache['time'], self.cache['rotor_speed4'], label='r_3')
        plt.legend()
        _ = plt.ylim()
        
    def plot_rotor_speed(self):
        plt.plot(self.cache['time'], self.cache['rotor_speed1'], label='Rotor 1 revolutions / second')
        plt.plot(self.cache['time'], self.cache['rotor_speed2'], label='Rotor 2 revolutions / second')
        plt.plot(self.cache['time'], self.cache['rotor_speed3'], label='Rotor 3 revolutions / second')
        plt.plot(self.cache['time'], self.cache['rotor_speed4'], label='Rotor 4 revolutions / second')
        plt.legend()
        _ = plt.ylim()