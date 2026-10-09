"""Shared VRX control and scene geometry; execution lives in run_experiment."""
import numpy as np
import torch

def force_to_thruster(force, phi, is_attacker=True, min_thrust=-500.0, max_thrust=1000.0):
    force = force / np.linalg.norm(force)
    rotation = np.array([
        [np.cos(phi), np.sin(phi)],
        [-np.sin(phi), np.cos(phi)]
    ])
    control = np.dot(rotation, force)
    acc = control[0]
    ang = np.arctan2(control[1], control[0])
    if is_attacker:
        l = (acc + 2.5 * ang) * max_thrust
        r = (acc - 2.5 * ang) * max_thrust
    else:
        l = (acc + 0.75 * ang) * max_thrust
        r = (acc - 0.75 * ang) * max_thrust
    return np.clip(np.array([l, r]), min_thrust, max_thrust)

def APF_navi_control(position, goal, obstacles, phi, min_thrust=-500.0, max_thrust=1000.0, improved=True):
    '''
        Artificial Potential Field Navigation
        
        Args:
            position: Robot's current position [x, y]
            goal: Robot's goal position [x, y]
            obstacles: List of obstacles' positions [[x1, y1], [x2, y2] ...]
            phi: Robot's heading
    '''
    if improved:
        alpha = 0.1
    else:
        alpha = 0.0

    k_att = 0.1
    goal_direction = goal - position
    att_force = k_att * goal_direction

    k_rep = 2000.0
    influence_radius = 20.0
    rep_force = np.zeros(2)
    for obs in obstacles:
        distance = max(np.linalg.norm(position - obs) - 7.0, 0.1)
        if distance < influence_radius:
            obs_direction = position - obs
            obs_direction = obs_direction / np.linalg.norm(obs_direction)
            perpendicular_direction = np.array([-obs_direction[1], obs_direction[0]])

            mag = k_rep * (1.0 / distance - 1.0 / influence_radius) / (distance ** 2)
            base_force = mag * obs_direction

            adjusted_force = (1 - alpha) * base_force + alpha * mag * perpendicular_direction
            rep_force += adjusted_force

            influence_radius = distance

    force = att_force + rep_force
    return force_to_thruster(force, phi, True, min_thrust, max_thrust)

def Boids_navi_control(positions, velocities, phis, goal, min_thrust=-500.0, max_thrust=1000.0):
    '''
        Boids Model Navigation
        
        Args:
            positions: USV swarm's current positions [num, 2]
            velocities: USV swarm's current velocities [num, 2]
            phis: USV swarm's headings 
    '''

    # Parameters
    neighbor_radius = 15.0

    k_att = 0.5
    k_sep = 10.0
    k_ali = 0.1
    k_coh = 0.1

    boids_forces = np.zeros_like(positions)
    boids_actions = np.zeros((positions.shape[0], 2))
    boids_states = np.zeros((positions.shape[0], positions.shape[1] * 3))
    for i, pos in enumerate(positions):
        dists = np.linalg.norm(positions - pos, axis=1)
        neighbors = dists < neighbor_radius
        neighbors[i] = False    # Exclude itself

        attraction = goal - pos

        seperation = np.zeros(2)
        for j, is_neighbor in enumerate(neighbors):
            if is_neighbor:
                diff = pos - positions[j]
                dist = max(np.linalg.norm(diff) - 3.5, 0.01)
                seperation += diff / (dist ** 2)
        
        alignment = np.mean(velocities, axis=0)

        cohesion = np.mean(positions, axis=0) - pos

        boids_forces[i] = k_att * attraction + k_sep * seperation + k_ali * alignment + k_coh * cohesion
        boids_states[i] = [*seperation, *alignment, *cohesion]
        boids_actions[i] = force_to_thruster(boids_forces[i], phis[i], False, min_thrust, max_thrust)

    return boids_actions, boids_states
    
def thrust_to_action(thrust:np.ndarray, min_thrust=-500.0, max_thrust=1000.0):
    action = (thrust * 2.0 - max_thrust - min_thrust) / (max_thrust - min_thrust)
    return np.clip(action, -1.0, 1.0)

def action_to_thrust(action:np.ndarray, min_thrust=-500.0, max_thrust=1000.0):
    action = np.clip(action, -1.0, 1.0)
    return (action * (max_thrust - min_thrust) + max_thrust + min_thrust) / 2.0

def RL_navi_control(actor, observations, boids_actions=None, controller='AdaRes',
                    device=torch.device('cpu'), min_thrust=-500.0, max_thrust=1000.0):
    with torch.no_grad():
        s = torch.tensor(observations, dtype=torch.float).to(device)
        a, _ = actor(s, True, False)
        a = a.cpu().numpy().flatten()
    actions = a.reshape(observations.shape[0], -1)
    if controller == 'Res':
        actions = action_to_thrust(actions, min_thrust, max_thrust) + boids_actions
    elif controller == 'AdaRes':
        adas = actions[:, 2]
        actions = action_to_thrust(np.copy(actions[:, :2]), min_thrust, max_thrust)
        for i in range(len(adas)):
            actions[i] = adas[i] * actions[i] + (1 - adas[i]) * boids_actions[i]
    elif controller == 'RL':
        actions = action_to_thrust(actions, min_thrust, max_thrust)
    else:
        raise ValueError('Unknown learned controller: '+controller)
    return np.clip(actions, min_thrust, max_thrust)

class ExperimentManager:
    def __init__(self, num_robots, device='cpu'):
        self.defend_r = 5.0
        self.collision_r = 5.0
        self.target_r = 15.0
        self.sensing_r = 60.0
        self.total_time = 60.0
        self.origin = np.array([-448.7194, 234.3858])
        self.num_robots = num_robots

        self.pose_data = {}

        self.curr_pos = np.zeros((num_robots, 2))
        self.curr_phi = np.zeros(num_robots)
        self.curr_vel = np.zeros((num_robots, 2))

        self.def_att_dists = np.zeros(self.num_robots - 1)
        self.def_def_dists = np.zeros((self.num_robots - 1, self.num_robots - 2))

        self.device = torch.device(device)

        for i in range(self.num_robots):
            self.pose_data[f'wamv{i+1}'] = []
        
    def generate_init_info(self, agility=2.0, setting=0):
        '''
            Generate initial positions

            Args:
                agility: attacker's agility level
                setting: 0 -> ocean setting (default); 
                         1 -> dock setting
        '''
        init_poses = ""
        self.agility = agility

        positions = np.zeros((self.num_robots - 1, 2))
        phis = np.zeros(self.num_robots - 1)
        goal = np.zeros(2)

        
        if setting == 0:
            self.origin = np.array([-448.7194, 234.3858])

            # Attacker init pose
            init_radius = np.random.uniform(self.sensing_r, self.sensing_r + 5)
            init_theta = np.random.uniform(0.0, 2 * np.pi)
            goal = init_radius * np.array([np.cos(init_theta), np.sin(init_theta)]) 
            init_pos = goal + self.origin
            self.curr_pos[0] = init_pos - self.origin

            init_poses += str(init_pos[0]) + ',' + str(init_pos[1]) + ',' + str(-init_theta)

            # Defenders init poses
            def_theta = np.random.uniform(-np.pi, np.pi)
            index = 2 * np.pi / (self.num_robots - 1)

            for i in range(self.num_robots - 1):
                radius = np.random.uniform(7.0, 8.0)
                theta = def_theta + index * i

                positions[i] = radius * np.array([np.cos(theta), np.sin(theta)])
                init_pos = positions[i] + self.origin
                self.curr_pos[i+1] = positions[i]

                phis[i] = init_theta + np.random.uniform(-np.pi / 3, np.pi / 3)
                init_poses += ';' + str(init_pos[0]) + ',' + str(init_pos[1]) + ',' + str(phis[i])
        
        elif setting == 1:
            self.origin = np.array([-576.2809, 270.6814])

            # Attacker init pose
            init_radius = np.random.uniform(self.sensing_r, self.sensing_r + 5)
            init_theta = np.random.uniform(0, np.pi / 4)
            goal = init_radius * np.array([np.cos(init_theta), np.sin(init_theta)]) 
            init_pos = goal + self.origin
            self.curr_pos[0] = init_pos - self.origin

            init_poses += str(init_pos[0]) + ',' + str(init_pos[1]) + ',' + str(-init_theta)

            # Defenders init poses
            def_theta = init_theta + np.pi / 3
            index = 2 * np.pi / (self.num_robots - 1)

            for i in range(self.num_robots - 1):
                radius = np.random.uniform(8.0, 9.0)
                theta = def_theta + index * i

                positions[i] = radius * np.array([np.cos(theta), np.sin(theta)])
                init_pos = positions[i] + self.origin
                self.curr_pos[i+1] = positions[i]

                phis[i] = init_theta + np.random.uniform(-np.pi / 3, np.pi / 3)
                init_poses += ';' + str(init_pos[0]) + ',' + str(init_pos[1]) + ',' + str(phis[i])

        boids_actions, boids_states = Boids_navi_control(positions, np.zeros((self.num_robots - 1, 2)), phis, goal)
        self.get_observations(positions, phis, goal, np.zeros(2), boids_states, boids_actions)
        
        return init_poses   


    


    def get_observations(self, positions, phis, goal, goal_vel, boids_states=None, boids_actions=None):
        def _calculate_dist_phi(vector, theta):
            dist = np.linalg.norm(vector)
            phi = np.arctan2(vector[1], vector[0]) - theta 
            if phi > np.pi:
                phi -= 2 * np.pi
            elif phi < -np.pi:
                phi += 2 * np.pi
            return dist, phi
        
        '''
            [d_Ti, phi_Ti, d_Ai, phi_Ai, v_A, phi_A, f_sep, phi_sep,
            f_ali, phi_ali, f_coh, phi_coh, a_boids, w_boids]
        '''
        robot_num = positions.shape[0]
        if boids_states is not None:
            assert boids_actions is not None, 'If using Boids states, you need Boids actions'
            feature_dim = 14
        else:
            feature_dim = 6
        observations = np.zeros((robot_num, feature_dim + 2 * (robot_num - 1)))
        for i, pos in enumerate(positions):
            theta = phis[i]
            observations[i, 0:2] = _calculate_dist_phi(-pos, theta)
            dist, phi = _calculate_dist_phi(goal - pos, theta)
            observations[i, 2:4] = dist, phi
            self.def_att_dists[i] = dist

            observations[i, 4:6] = _calculate_dist_phi(goal_vel, theta)

            if boids_states is not None:
                observations[i, 6:8] = _calculate_dist_phi(boids_states[i, 0:2], theta)
                observations[i, 8:10] = _calculate_dist_phi(boids_states[i, 2:4], theta)
                observations[i, 10:12] = _calculate_dist_phi(boids_states[i, 4:6], theta)
                observations[i, 12:14] = thrust_to_action(boids_actions[i])
            
            k=0
            for j, teammate_pos in enumerate(positions):
                if j == i:
                    continue
                else:
                    dist, phi = _calculate_dist_phi(teammate_pos - pos, theta)
                    observations[i, feature_dim+2*k:feature_dim+2+2*k] = dist, phi
                    self.def_def_dists[i, k] = dist
                    k += 1
            
        return observations


if __name__ == "__main__":
    # Keep the published command usable with the validated simulation runner.
    from run_experiment import main
    raise SystemExit(main())
