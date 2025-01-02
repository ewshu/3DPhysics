import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
import random
from enum import Enum


class ParticleType(Enum):
    NORMAL = 1
    ATTRACTOR = 2
    REPULSOR = 3
    NEUTRAL = 4


class Particle:
    def __init__(self, position, velocity, mass, particle_type=ParticleType.NORMAL):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.charge = random.uniform(-1, 1) if particle_type != ParticleType.NEUTRAL else 0
        self.particle_type = particle_type
        self.trail = []
        self.trail_max_length = 50
        self.energy = 0.5 * mass * np.dot(velocity, velocity)  # Kinetic energy
        self.creation_time = glutGet(GLUT_ELAPSED_TIME)

    def update_trail(self):
        self.trail.append(np.copy(self.position))
        if len(self.trail) > self.trail_max_length:
            self.trail.pop(0)

    def get_color(self):
        # Color based on velocity and particle type
        speed = np.linalg.norm(self.velocity)
        if self.particle_type == ParticleType.ATTRACTOR:
            return (min(1.0, speed * 0.2), 0.2, 0.2)
        elif self.particle_type == ParticleType.REPULSOR:
            return (0.2, 0.2, min(1.0, speed * 0.2))
        else:
            return (0.2, min(1.0, speed * 0.2), 0.2)


class ParticleSimulator:
    def __init__(self, width=1200, height=800):
        # Initialize GLUT
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        glutInitWindowPosition(100, 100)
        glutCreateWindow(b"Enhanced 3D Particle Physics Simulator")

        # Set callbacks
        glutDisplayFunc(self.display)
        glutIdleFunc(self.idle)
        glutReshapeFunc(self.reshape)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.motion)
        glutKeyboardFunc(self.keyboard)
        glutSpecialFunc(self.special_keys)

        # Initialize parameters
        self.width = width
        self.height = height
        self.running = False
        self.show_trails = False
        self.show_vectors = False
        self.show_forces = False
        self.show_grid = True
        self.current_particle_type = ParticleType.NORMAL

        # Physics parameters
        self.gravity_strength = 0.1
        self.electromagnetic_strength = 0.05
        self.temperature = 1.0
        self.damping = 0.999

        # Camera parameters
        self.camera_distance = 50.0
        self.camera_rotation = [0.0, 0.0]
        self.last_mouse_pos = None

        # Particle system
        self.particles = []
        self.create_particles(10)

        # UI state
        self.selected_particle = None
        self.mouse_world_pos = np.zeros(3)
        self.creation_mode = False

        # Set up OpenGL scene
        self.setup_scene()
        self.last_update = glutGet(GLUT_ELAPSED_TIME)

    def create_particles(self, num_particles):
        for _ in range(num_particles):
            position = [random.uniform(-20, 20) for _ in range(3)]
            velocity = [random.uniform(-1, 1) for _ in range(3)]
            mass = random.uniform(0.5, 2.5)
            p_type = random.choice(list(ParticleType))
            self.particles.append(Particle(position, velocity, mass, p_type))

    def setup_scene(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Set up light
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])

    def calculate_forces(self, particle, other):
        r = other.position - particle.position
        distance = np.linalg.norm(r)
        if distance < 0.1:
            return np.zeros(3)

        # Gravitational force
        f_gravity = self.gravity_strength * particle.mass * other.mass / (distance ** 2)

        # Electromagnetic force
        f_em = self.electromagnetic_strength * particle.charge * other.charge / (distance ** 2)

        # Particle type specific forces
        type_factor = 1.0
        if particle.particle_type == ParticleType.ATTRACTOR:
            type_factor = 2.0
        elif particle.particle_type == ParticleType.REPULSOR:
            type_factor = -1.0

        total_force = (f_gravity + f_em) * type_factor
        return total_force * r / distance

    def update_physics(self):
        if not self.running:
            return

        current_time = glutGet(GLUT_ELAPSED_TIME)
        dt = (current_time - self.last_update) / 1000.0
        self.last_update = current_time

        # Update particles
        for i, particle in enumerate(self.particles):
            total_force = np.zeros(3)

            # Calculate forces from other particles
            for j, other in enumerate(self.particles):
                if i != j:
                    force = self.calculate_forces(particle, other)
                    total_force += force

            # Apply temperature effects
            total_force += np.random.normal(0, self.temperature, 3)

            # Update velocity and position
            acceleration = total_force / particle.mass
            particle.velocity += acceleration * dt
            particle.velocity *= self.damping  # Apply damping
            particle.position += particle.velocity * dt

            # Update energy
            particle.energy = 0.5 * particle.mass * np.dot(particle.velocity, particle.velocity)

            # Boundary conditions (bounce off walls)
            for axis in range(3):
                if abs(particle.position[axis]) > 25:
                    particle.position[axis] = np.sign(particle.position[axis]) * 25
                    particle.velocity[axis] *= -0.8

            # Update particle trail
            if self.show_trails:
                particle.update_trail()

            # Remove particles that are too old or have too little energy
            if particle.energy < 0.01 or current_time - particle.creation_time > 30000:
                self.particles.remove(particle)

    def draw_vector(self, origin, vector, color, scale=1.0):
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        glColor3f(*color)
        glVertex3fv(origin)
        end_point = origin + vector * scale
        glVertex3fv(end_point)

        # Draw arrow head
        if np.linalg.norm(vector) > 0:
            arrow_length = 0.5
            direction = vector / np.linalg.norm(vector)
            up = np.array([0, 1, 0])
            if abs(np.dot(direction, up)) < 0.999:
                right = np.cross(direction, up)
                right = right / np.linalg.norm(right)
                up = np.cross(right, direction)
            else:
                right = np.array([1, 0, 0])
                up = np.cross(right, direction)

            for i in range(4):
                angle = i * math.pi / 2
                arrow_vec = (right * math.cos(angle) + up * math.sin(angle)) * arrow_length
                glVertex3fv(end_point)
                glVertex3fv(end_point - direction * arrow_length + arrow_vec)

        glEnd()
        glEnable(GL_LIGHTING)

    def draw_grid(self):
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 0.2, 0.2)
        glBegin(GL_LINES)
        for i in range(-25, 26, 5):
            glVertex3f(i, -25, 0)
            glVertex3f(i, 25, 0)
            glVertex3f(-25, i, 0)
            glVertex3f(25, i, 0)
        glEnd()
        glEnable(GL_LIGHTING)

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Set up camera
        gluLookAt(
            self.camera_distance * math.sin(self.camera_rotation[0]) * math.cos(self.camera_rotation[1]),
            self.camera_distance * math.sin(self.camera_rotation[1]),
            self.camera_distance * math.cos(self.camera_rotation[0]) * math.cos(self.camera_rotation[1]),
            0, 0, 0,
            0, 1, 0
        )

        # Draw grid
        if self.show_grid:
            self.draw_grid()

        # Draw particles
        for particle in self.particles:
            # Draw particle
            glPushMatrix()
            glTranslatef(*particle.position)
            glColor3f(*particle.get_color())
            glutSolidSphere(particle.mass * 0.5, 16, 16)
            glPopMatrix()

            # Draw trail
            if self.show_trails and len(particle.trail) > 1:
                glDisable(GL_LIGHTING)
                glColor4f(*particle.get_color(), 0.5)
                glBegin(GL_LINE_STRIP)
                for point in particle.trail:
                    glVertex3fv(point)
                glEnd()
                glEnable(GL_LIGHTING)

            # Draw velocity vector
            if self.show_vectors:
                self.draw_vector(particle.position, particle.velocity, (1, 1, 0), 2.0)

            # Draw force vectors
            if self.show_forces:
                total_force = np.zeros(3)
                for other in self.particles:
                    if other != particle:
                        force = self.calculate_forces(particle, other)
                        total_force += force
                self.draw_vector(particle.position, total_force, (1, 0, 1), 0.5)

        # Draw UI
        self.draw_ui()

        glutSwapBuffers()

    def draw_ui(self):
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Draw status and controls
        y = 20
        for text in [
            f"Status: {'Running' if self.running else 'Paused'}",
            f"Particle Type: {self.current_particle_type.name}",
            f"Particles: {len(self.particles)}",
            f"Gravity: {self.gravity_strength:.2f}",
            f"Temperature: {self.temperature:.2f}",
            "Controls:",
            "Space - Start/Pause",
            "T - Toggle Trails",
            "V - Toggle Velocity Vectors",
            "F - Toggle Force Vectors",
            "G - Toggle Grid",
            "Click - Create Particle",
            "1-4 - Select Particle Type",
            "Up/Down - Adjust Gravity",
            "+/- - Adjust Temperature",
            "Esc - Quit"
        ]:
            self.render_text(10, y, text)
            y += 20

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_LIGHTING)

    def render_text(self, x, y, text):
        glColor3f(0.0, 1.0, 0.0)
        glRasterPos2f(x, y)
        for char in text:
            glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ord(char))

    def idle(self):
        self.update_physics()
        glutPostRedisplay()

    def screen_to_world(self, screen_x, screen_y):
        # Get the current viewport
        viewport = glGetIntegerv(GL_VIEWPORT)

        # Get the modelview and projection matrices
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)

        # Get the window coordinates of the mouse click
        win_x = float(screen_x)
        win_y = float(viewport[3] - screen_y)

        # Get the depth buffer value at the mouse position
        win_z = glReadPixels(int(win_x), int(win_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]

        # Unproject the point
        world_x, world_y, world_z = gluUnProject(win_x, win_y, win_z, modelview, projection, viewport)

        return np.array([world_x, world_y, world_z])

    def mouse(self, button, state, x, y):
        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self.last_mouse_pos = (x, y)
                if glutGetModifiers() & GLUT_ACTIVE_CTRL:
                    # Create new particle
                    world_pos = self.screen_to_world(x, y)
                    velocity = np.random.uniform(-1, 1, 3)
                    mass = random.uniform(0.5, 2.5)
                    self.particles.append(Particle(world_pos, velocity, mass, self.current_particle_type))
            else:
                self.last_mouse_pos = None
        elif state == GLUT_UP:
            if button == 3:  # Mouse wheel up
                self.camera_distance *= 0.9
            elif button == 4:  # Mouse wheel down
                self.camera_distance *= 1.1
            self.camera_distance = max(10, min(self.camera_distance, 100))

    def motion(self, x, y):
        if self.last_mouse_pos:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            self.camera_rotation[0] += dx * 0.01
            self.camera_rotation[1] += dy * 0.01
            self.camera_rotation[1] = max(min(self.camera_rotation[1], math.pi / 2), -math.pi / 2)
            self.last_mouse_pos = (x, y)

    def keyboard(self, key, x, y):
        key = key.decode('utf-8').lower()
        if key == ' ':
            self.running = not self.running
            self.last_update = glutGet(GLUT_ELAPSED_TIME)
        elif key == 't':
            self.show_trails = not self.show_trails
            if not self.show_trails:
                for particle in self.particles:
                    particle.trail.clear()
        elif key == 'v':
            self.show_vectors = not self.show_vectors
        elif key == 'f':
            self.show_forces = not self.show_forces
        elif key == 'g':
            self.show_grid = not self.show_grid
        elif key in ['1', '2', '3', '4']:
            self.current_particle_type = ParticleType(int(key))
        elif key == '+':
            self.temperature = min(2.0, self.temperature + 0.1)
        elif key == '-':
            self.temperature = max(0.0, self.temperature - 0.1)
        elif key == '\x1b':  # ESC key
            sys.exit(0)

    def special_keys(self, key, x, y):
        if key == GLUT_KEY_UP:
            self.gravity_strength = min(1.0, self.gravity_strength + 0.05)
        elif key == GLUT_KEY_DOWN:
            self.gravity_strength = max(0.0, self.gravity_strength - 0.05)

    def reshape(self, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def run(self):
        glutMainLoop()


if __name__ == "__main__":
    simulator = ParticleSimulator()
    simulator.run()