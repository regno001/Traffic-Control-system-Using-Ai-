import tkinter as tk
from tkinter import messagebox
import threading
import time


class TrafficLightApp:
    def __init__(self, master):
        self.master = master
        master.title("AI Traffic Control System")
        master.configure(bg="#f0f0f0")

        self.lane_signals = {}
        self.simulation_running = False
        self.simulation_thread = None

        self.create_widgets()

    def create_widgets(self):
        # Frame for input controls
        input_frame = tk.Frame(self.master, padx=10, pady=10, bg="#f0f0f0")
        input_frame.pack(pady=10)

        # Number of Lanes input
        tk.Label(input_frame, text="Number of Lanes:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.num_lanes_entry = tk.Entry(input_frame, width=5)
        self.num_lanes_entry.pack(side=tk.LEFT, padx=5)

        # Start/Stop button
        self.start_button = tk.Button(input_frame, text="Start Simulation", command=self.start_stop_simulation,
                                      bg="#4CAF50", fg="white", activebackground="#45a049")
        self.start_button.pack(side=tk.LEFT, padx=10)

        # ---
        # Frame for emergency controls
        emergency_frame = tk.Frame(self.master, padx=10, pady=10, bg="#f0f0f0")
        emergency_frame.pack(pady=5)

        # Ambulance input
        tk.Label(emergency_frame, text="Ambulance on Lane:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.ambulance_lane_entry = tk.Entry(emergency_frame, width=5)
        self.ambulance_lane_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(emergency_frame, text="Ambulance Arrives", command=self.handle_ambulance, bg="#2196F3", fg="white",
                  activebackground="#1976D2").pack(side=tk.LEFT, padx=5)

        # ---
        # Frame for accident and violator controls
        incident_frame = tk.Frame(self.master, padx=10, pady=10, bg="#f0f0f0")
        incident_frame.pack(pady=5)

        # Accident input
        tk.Label(incident_frame, text="Accident on Lane:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.accident_lane_entry = tk.Entry(incident_frame, width=5)
        self.accident_lane_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(incident_frame, text="Report Accident", command=self.report_accident, bg="#f44336", fg="white",
                  activebackground="#d32f2f").pack(side=tk.LEFT, padx=5)

        # Violator input
        tk.Label(incident_frame, text="Violator on Lane:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.violator_lane_entry = tk.Entry(incident_frame, width=5)
        self.violator_lane_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(incident_frame, text="Report Violator", command=self.report_violator, bg="#FFC107", fg="white",
                  activebackground="#FFB300").pack(side=tk.LEFT, padx=5)

        # ---
        # Frame for traffic signals
        self.signal_frame = tk.Frame(self.master, bg="#f0f0f0")
        self.signal_frame.pack(pady=20)

    def start_stop_simulation(self):
        if self.simulation_running:
            self.stop_simulation()
        else:
            try:
                num_lanes = int(self.num_lanes_entry.get())
                if num_lanes > 0:
                    self.setup_signals(num_lanes)
                    self.simulation_running = True
                    self.start_button.config(text="Stop Simulation", bg="#d32f2f")
                    self.simulation_thread = threading.Thread(target=self.run_simulation, args=(num_lanes,))
                    self.simulation_thread.daemon = True
                    self.simulation_thread.start()
                else:
                    messagebox.showerror("Invalid Input", "Please enter a positive integer for the number of lanes.")
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number.")

    def stop_simulation(self):
        if self.simulation_running:
            self.simulation_running = False
            self.start_button.config(text="Start Simulation", bg="#4CAF50")
            # Wait for the thread to finish
            if self.simulation_thread and self.simulation_thread.is_alive():
                messagebox.showinfo("Stopping", "Stopping simulation. Please wait...")

    def setup_signals(self, num_lanes):
        # Clear existing signals
        for widget in self.signal_frame.winfo_children():
            widget.destroy()
        self.lane_signals = {}

        # Create signals for each lane
        for i in range(1, num_lanes + 1):
            lane_frame = tk.Frame(self.signal_frame, padx=10, pady=10, bg="#f0f0f0")
            lane_frame.pack(side=tk.LEFT, padx=10)

            tk.Label(lane_frame, text=f"Lane {i}", font=("Arial", 12, "bold"), bg="#f0f0f0").pack()

            # Create a canvas for the traffic light
            canvas = tk.Canvas(lane_frame, width=50, height=150, bg="#333", highlightthickness=0, bd=0)
            canvas.pack()

            # Draw the lights (circles)
            red_light = canvas.create_oval(10, 10, 40, 40, fill="#555")
            yellow_light = canvas.create_oval(10, 60, 40, 90, fill="#555")
            green_light = canvas.create_oval(10, 110, 40, 140, fill="#555")

            self.lane_signals[i] = {
                'canvas': canvas,
                'lights': [red_light, yellow_light, green_light]
            }
            self.update_signal(i, 'red')

    def update_signal(self, lane, color):
        if lane in self.lane_signals:
            canvas = self.lane_signals[lane]['canvas']
            lights = self.lane_signals[lane]['lights']

            # Reset all lights
            for light in lights:
                canvas.itemconfig(light, fill="#555")

            # Set the color for the active light
            if color == 'red':
                canvas.itemconfig(lights[0], fill="red")
            elif color == 'yellow':
                canvas.itemconfig(lights[1], fill="yellow")
            elif color == 'green':
                canvas.itemconfig(lights[2], fill="green")
            self.master.update_idletasks()

    def run_simulation(self, num_lanes):
        if num_lanes == 4:
            lane_groups = [[1, 3], [2, 4]]
        elif num_lanes == 3:
            lane_groups = [[1, 3], [2]]
        else:
            lane_groups = [[lane] for lane in range(1, num_lanes + 1)]

        while self.simulation_running:
            for group in lane_groups:
                if not self.simulation_running:
                    break

                other_lanes = [lane for lane in range(1, num_lanes + 1) if lane not in group]

                # Green for current group
                for lane in group:
                    self.update_signal(lane, 'green')

                # Red for others
                for lane in other_lanes:
                    self.update_signal(lane, 'red')

                time.sleep(10)  # Simulating 3 minutes (180 seconds) for demonstration
                if not self.simulation_running:
                    break

                # Yellow transition
                for lane in group:
                    self.update_signal(lane, 'yellow')

                time.sleep(3)  # Simulating 30 seconds for demonstration
                if not self.simulation_running:
                    break

    def handle_ambulance(self):
        try:
            lane = int(self.ambulance_lane_entry.get())
            if 1 <= lane <= len(self.lane_signals):
                if self.simulation_running:
                    self.stop_simulation()

                all_lanes = list(self.lane_signals.keys())
                other_lanes = [l for l in all_lanes if l != lane]

                # Set green for ambulance lane, red for others
                self.update_signal(lane, 'green')
                for l in other_lanes:
                    self.update_signal(l, 'red')

                messagebox.showinfo("Emergency Alert",
                                    f"Ambulance has arrived on Lane {lane}. All other lanes have been cleared.")

                # Resume normal simulation after a delay
                self.master.after(5000, self.resume_simulation)
            else:
                messagebox.showerror("Invalid Lane", "Please enter a valid lane number.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")

    def resume_simulation(self):
        self.simulation_running = True
        self.start_button.config(text="Stop Simulation", bg="#d32f2f")
        self.simulation_thread = threading.Thread(target=self.run_simulation, args=(len(self.lane_signals),))
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def report_accident(self):
        try:
            lane = int(self.accident_lane_entry.get())
            if 1 <= lane <= len(self.lane_signals):
                messagebox.showinfo("Accident Alert",
                                    f"An accident has been detected on Lane {lane}. Calling emergency services: 108 (Ambulance) and 100 (Police).")
            else:
                messagebox.showerror("Invalid Lane", "Please enter a valid lane number for the accident.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")

    def report_violator(self):
        try:
            lane = int(self.violator_lane_entry.get())
            if 1 <= lane <= len(self.lane_signals):
                messagebox.showinfo("Violation Alert",
                                    f"A traffic violation has been detected on Lane {lane}. Image captured and stored for review.")
            else:
                messagebox.showerror("Invalid Lane", "Please enter a valid lane number for the violator.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")


def main():
    root = tk.Tk()
    app = TrafficLightApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
