from networkx import grid_2d_graph, relabel_nodes
from networkx.algorithms.approximation.steinertree import steiner_tree
from itertools import product
from math import ceil
from functools import lru_cache
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import csv, json

root = tk.Tk()
root.withdraw()

SHOW_FIGURES = messagebox.askyesno(
	title="Show Figures",
	message="Do you want to display figures of generated routes? (this can generate many browser windows)"
	)
SAVE_FIGURES = messagebox.askyesno(
	title="Save Figures",
	message="Do you want to save figures of generated routes? (this can take a long time)"
	)
if SAVE_FIGURES == True:
	figures_dir = Path(
		filedialog.askdirectory(
			title="Select Figures Folder"
			)
		)

class WireType:

	cmils = {'#12':6530, '#10':10380, '#8':16510, '#6':26240, '#4':41740, '#3':52660, '#2':66360, '#1':83690, '1/0':105600, '2/0':133100, '3/0':167800, '4/0':211600, '250':250000, '300':300000, '350':350000, '400':400000, '500':500000, '600':600000, '700':700000, '750':750000, '800':800000, '900':900000, '1000':1000000, '1250':1250000, '1500':1500000, '1750':1750000, '2000':2000000}

	amps = {'Cu': {
					60: {'#12':25, '#10':30, '#8':40, '#6':55, '#4':70, '#3':85, '#2':95, '#1':110, '1/0':125, '2/0':145, '3/0':165, '4/0':195, '250':215, '300':240, '350':260, '400':280, '500':320, '600':355, '700':385, '750':400, '800':410, '900':435, '1000':455, '1250':495, '1500':520, '1750':545, '2000':665},
					75:{'#12':25, '#10':35, '#8':50, '#6':65, '#4':85, '#3':100, '#2':115, '#1':130, '1/0':150, '2/0':175, '3/0':200, '4/0':230, '250':255, '300':285, '350':310, '400':355, '500':380, '600':420, '700':460, '750':475, '800':490, '900':520, '1000':545, '1250':590, '1500':625, '1750':650, '2000':665},
					90:{'#12':30, '#10':40, '#8':55, '#6':75, '#4':95, '#3':110, '#2':130, '#1':150, '1/0':170, '2/0':195, '3/0':225, '4/0':260, '250':290, '300':320, '350':350, '400':380, '500':430, '600':475, '700':520, '750':535, '800':555, '900':585, '1000':615, '1250':665, '1500':705, '1750':735, '2000':750} } ,
			'Al': {
					60:{'#12':20, '#10':25, '#8':30, '#6':40, '#4':55, '#3':65, '#2':75, '#1':85, '1/0':100, '2/0':115, '3/0':130, '4/0':150, '250':170, '300':190, '350':210, '400':225, '500':260, '600':285, '700':310, '750':320, '800':330, '900':355, '1000':375, '1250':405, '1500':435, '1750':455, '2000':470} ,
					75:{'#12':20, '#10':30, '#8':40, '#6':50, '#4':65, '#3':75, '#2':90, '#1':100, '1/0':120, '2/0':135, '3/0':155, '4/0':180, '250':205, '300':230, '350':250, '400':270, '500':310, '600':340, '700':375, '750':385, '800':395, '900':425, '1000':445, '1250':485, '1500':520, '1750':545, '2000':560} ,
					90:{'#12':25, '#10':35, '#8':45, '#6':60, '#4':75, '#3':85, '#2':100, '#1':115, '1/0':135, '2/0':150, '3/0':175, '4/0':205, '250':230, '300':255, '350':280, '400':305, '500':350, '600':385, '700':420, '750':435, '800':450, '900':480, '1000':500, '1250':545, '1500':585, '1750':615, '2000':630} } }

	grnd = {'Cu': {
					'#12':20, '#10':60, '#8':100, '#6':200, '#4':300, '#3':400, '#2':500, '#1':600, '1/0':800, '2/0':1000, '3/0':1200, '4/0':1600, '250':2000} ,
			'Al': {
					'#12':15, '#10':20, '#8':60, '#6':100, '#4':200, '#2':300, '#1':400, '1/0':500, '2/0':600, '3/0':800, '4/0':1000, '250':1200, '350':1600, '400':2000} }

	ohms = {'Cu':12.9, 'Al':21.2}

	def __init__(self,material,temp):
		self.material = material
		self.temp = temp
		self.amps = self.amps[material][temp]
		self.ohms = self.ohms[material]
		self.grnd = self.grnd[material]

THHN = WireType('Cu',90)

class elecDevice:

	def __init__(self, location, amps=0):
		self.location = location
		self.amps = amps

class elecPanel:

	def __init__(self, name, voltage, location, phases=3):
		self.name = name
		self.phases = phases
		self.voltage = voltage
		self.location = location
		self.circuits = {}

		if voltage == 480:
			self.colors = ['yellow','brown','orange','grey','green'] # self.colors.index(circuit_color) == REMAINDER(ciruit_number / 3)
		elif voltage == 208:
			self.colors = ['blue','black','red','white','green'] # self.colors.index(circuit_color) == REMAINDER(ciruit_number / 3)
	
	@lru_cache(maxsize=None)
	def getCircuitColor(self,circ_num):
		color_num = ceil(circ_num/2)%3
		color = self.colors[color_num]
		return color
	
	@lru_cache(maxsize=None)
	def getTotalWire(self):
		total_wire_array = sum( self.circuits[circuit].getTotalWire(format='array') for circuit in self.circuits )
		return total_wire_array

class elecCircuit:
	
	def __init__(self, panel, circ_nums, trip_amps, run_elev, devices, wire_type=THHN, derating=0.8):
		self.panel = panel
		self.circ_nums = circ_nums
		self.trip_amps = trip_amps
		self.run_elev = run_elev
		self.devices = devices
		self.wire_type = wire_type		
		self.dev_amps = sum([d.amps for d in self.devices])
		self.derating = derating
		self.conductors = []
		
		if len(self.circ_nums) == 1:
			self.voltage = round(self.panel.voltage/(3**(1/2)))
			self.phases = 1
			self.conductors.append(self.panel.getCircuitColor(self.circ_nums[0])) # hot color
			self.conductors.append(self.panel.colors[3]) # neutral color
			self.conductors.append(self.panel.colors[4]) # ground color
		elif len(self.circ_nums) == 2:
			self.voltage = self.panel.voltage
			self.phases = 1
			self.conductors.append(self.panel.getCircuitColor(self.circ_nums[0])) # hot 1 color
			self.conductors.append(self.panel.getCircuitColor(self.circ_nums[1])) # hot 2 color
			self.conductors.append(self.panel.colors[4]) # ground color
		elif len(self.circ_nums) == 3:
			self.voltage = self.panel.voltage
			self.phases = 3
			self.conductors.append(self.panel.getCircuitColor(self.circ_nums[0])) # hot 1 color
			self.conductors.append(self.panel.getCircuitColor(self.circ_nums[1])) # hot 2 color
			self.conductors.append(self.panel.getCircuitColor(self.circ_nums[2])) # hot 3 color
			self.conductors.append(self.panel.colors[3]) # neutral color
			self.conductors.append(self.panel.colors[4]) # ground color

	@lru_cache(maxsize=None)
	def drawGraph(self,graph):
		nodes = graph.nodes
		edges = graph.edges
		nX = []
		nY = []
		eX = []
		eY = []
		for n in nodes:
			nX.append(n[0])
			nY.append(n[1])
		for e in edges:
			eX.append(e[0][0])
			eX.append(e[1][0])
			eX.append(None)
			eY.append(e[0][1])
			eY.append(e[1][1])
			eY.append(None)
		nodePlot = go.Scatter(x=nX, y=nY, mode='markers')
		edgePlot = go.Scatter(x=eX, y=eY, mode='lines')
		fig = go.Figure(data=[nodePlot,edgePlot])
		if SHOW_FIGURES == True:
			fig.show()
		if SAVE_FIGURES == True:
			fig.write_image(
				figures_dir.joinpath(
					f'{str(self.panel.name)}{str(self.circ_nums)}.svg'
					)
				)

	@lru_cache(maxsize=None)
	def getGraph(self):
		devX = [self.panel.location[0]] + [d.location[0] for d in self.devices]
		devY = [self.panel.location[1]] + [d.location[1] for d in self.devices]
		nodeX = []
		nodeY = []
		for i in devX:
			if i not in nodeX:
				nodeX.append(i)
		for j in devY:
			if j not in nodeY:
				nodeY.append(j)
		graph = grid_2d_graph(len(nodeX),len(nodeY))
		graph.graph['term_nodes'] = []
		relabel = {}
		for p in product(enumerate(nodeX),enumerate(nodeY)):
			x0 = p[0][0]
			x1 = p[0][1]
			y0 = p[1][0]
			y1 = p[1][1]
			relabel[(x0,y0)] = (x1,y1)
		for x,y in zip(devX, devY):
			graph.graph['term_nodes'].append((x,y))
		graph = relabel_nodes(graph,relabel)
		#self.drawGraph(graph)
		return graph

	@lru_cache(maxsize=None)
	def getSteiner(self):
		graph = self.getGraph()
		steiner = steiner_tree(graph, graph.graph['term_nodes'])
		for e in steiner.edges:
			steiner.edges[e]['length'] = abs(e[0][0]-e[1][0]) + abs(e[0][1]-e[1][1])
		if SHOW_FIGURES == True or SAVE_FIGURES == True:
			self.drawGraph(steiner)
		return steiner
	
	@lru_cache(maxsize=None)
	def getLength(self):
		length = sum( [self.getSteiner().edges[e]['length'] for e in self.getSteiner().edges ] ) + 2*sum( [ abs(self.run_elev-d.location[2]) for d in self.devices ] ) + abs(self.run_elev-self.panel.location[2])
		return length
	
	@lru_cache(maxsize=None)
	def getBaseWireSize(self):
		keys = list(self.wire_type.amps.keys())
		vals = list(self.wire_type.amps.values())
		revVals = list(reversed(vals))
		wire_size_max_amps = 0
		while wire_size_max_amps < self.trip_amps:
			wire_size_max_amps = revVals.pop()
		base_wire_size = keys[vals.index(wire_size_max_amps)]
		return base_wire_size
	
	@lru_cache(maxsize=None)
	def getDeratedWireSize(self):
		keys = list(self.wire_type.amps.keys())
		vals = list(self.wire_type.amps.values())
		revVals = list(reversed(vals))
		wire_size_max_amps = 0
		while self.derating*wire_size_max_amps < self.trip_amps:
			wire_size_max_amps = revVals.pop()
		derated_wire_size = keys[vals.index(wire_size_max_amps)]
		return derated_wire_size

	@lru_cache(maxsize=None)
	def getGroundWireSize(self):
		keys = list(self.wire_type.grnd.keys())
		vals = list(self.wire_type.grnd.values())
		revVals = list(reversed(vals))
		wire_size_max_amps = 0
		while wire_size_max_amps < self.trip_amps:
			wire_size_max_amps = revVals.pop()
		ground_wire_size = keys[vals.index(wire_size_max_amps)]
		return ground_wire_size
	
	@lru_cache(maxsize=None)
	def getVoltageDropWireLength(self,wire_size):
		phase_mult = {1:2, 3:3**(1/2)}
		v = self.voltage
		i = self.trip_amps	
		p = phase_mult[self.phases]
		k = self.wire_type.ohms
		c = self.wire_type.cmils[wire_size]
		max_wire_length = (0.03 * v * c) / (p * k * i)
		return max_wire_length
	
	@lru_cache(maxsize=None)
	def getTotalWireLengthsLinear(self):
		# need to adjust this to account for TOTAL voltage drop, and discrete length SEGMENTS
		total_wire_lengths = {}
		def getNextWireLength(self, wire_size):
			wire_sizes = list(self.wire_type.cmils.keys())
			if sum(list(total_wire_lengths.values())) + self.getVoltageDropWireLength(wire_size) >= self.getLength():
				total_wire_lengths[wire_size] = self.getLength() - sum(list(total_wire_lengths.values()))
			else:
				total_wire_lengths[wire_size] = self.getVoltageDropWireLength(wire_size)
				getNextWireLength(self,wire_sizes[wire_sizes.index(wire_size)+1])
		getNextWireLength(self,self.getDeratedWireSize())
		return total_wire_lengths

	@lru_cache(maxsize=None)
	def getTotalWireLengths(self):
		# need to adjust this to account for TOTAL voltage drop, and discrete length SEGMENTS
		total_wire_lengths = {}
		wire_sizes = list(self.wire_type.cmils.keys())
		cur_wire_size = self.getDeratedWireSize()
		next_wire_size = wire_sizes[ wire_sizes.index(cur_wire_size)+1 ]
		if self.getVoltageDropWireLength( cur_wire_size ) < self.getLength():
			total_wire_lengths[ next_wire_size ] = self.getVoltageDropWireLength( next_wire_size )
			total_wire_lengths[ cur_wire_size ] = self.getLength() - self.getVoltageDropWireLength( next_wire_size )
		else:
			total_wire_lengths[ cur_wire_size ] = self.getLength()
		return total_wire_lengths

	@lru_cache(maxsize=None)
	def getTotalWire(self,format='dict'):
		if len(self.devices)==1:
			if format == 'dict':
				total_wire_dict = {}		
				for color in self.conductors[:-1]:
					total_wire_dict[color] = self.getTotalWireLengthsLinear()
				total_wire_dict['green'] = {self.getGroundWireSize() : round(self.getLength())}
				return total_wire_dict
			elif format == 'array':
				A = self.panel.colors[1]
				B = self.panel.colors[2]
				C = self.panel.colors[0]
				N = self.panel.colors[3]
				G = self.panel.colors[4]
				ordered_color_list = [A,B,C,N,G]
				total_wire_array = np.zeros( ( len(ordered_color_list), len(list(WireType.cmils)) ) )
				for color,size in product( self.conductors[:-1], self.getTotalWireLengthsLinear() ):
					coords = ( ordered_color_list.index(color), list(WireType.cmils).index(size) )
					total_wire_array[coords] = self.getTotalWireLengthsLinear()[ list(WireType.cmils)[coords[1]] ]
				total_wire_array[ ( ordered_color_list.index('green'), list(WireType.cmils).index(self.getGroundWireSize()) ) ] = round(self.getLength())
				return total_wire_array
		else:
			if format == 'dict':
				total_wire_dict = {}		
				for color in self.conductors[:-1]:
					total_wire_dict[color] = self.getTotalWireLengths()
				total_wire_dict['green'] = {self.getGroundWireSize() : round(self.getLength())}
				return total_wire_dict
			elif format == 'array':
				A = self.panel.colors[1]
				B = self.panel.colors[2]
				C = self.panel.colors[0]
				N = self.panel.colors[3]
				G = self.panel.colors[4]
				ordered_color_list = [A,B,C,N,G]
				total_wire_array = np.zeros(( len(ordered_color_list), len(list(WireType.cmils)) ))
				for color,size in product( self.conductors[:-1], self.getTotalWireLengths() ):
					coords = ( ordered_color_list.index(color), list(WireType.cmils).index(size) )
					total_wire_array[coords] = self.getTotalWireLengths()[ list(WireType.cmils)[coords[1]] ]
				total_wire_array[ ( ordered_color_list.index('green'), list(WireType.cmils).index(self.getGroundWireSize()) ) ] = round(self.getLength())
				return total_wire_array

json_path = filedialog.askopenfilename(title="Select Panel Data JSON File")

with open(json_path,'r') as json_file:
	panel_data = json.load(json_file)

panels = {}

for panel in panel_data:
	panels[panel] = elecPanel(
			name=panel,
			voltage=panel_data[panel]['voltage'],
			location=panel_data[panel]['location'],
			phases=3
			)
	
	for circuit in panel_data[panel]['circuits']:
		panels[panel].circuits[circuit] = elecCircuit(
				panel = panels[panel],
				circ_nums = panel_data[panel]['circuits'][circuit]['circ_nums'],
				trip_amps = panel_data[panel]['circuits'][circuit]['trip_amps'],
				run_elev = panel_data[panel]['circuits'][circuit]['run_elev'],
				devices = [
						elecDevice(location = i)
						for i in list(panel_data[panel]['circuits'][circuit]['devices'].values())
						]
				)

csv_dir = Path(filedialog.askdirectory(title="Select Wire Data CSVs Folder"))

for panel in panels:
	for circuit in panels[panel].circuits:
		print(panel,circuit,':')
		for color in panels[panel].circuits[circuit].getTotalWire():
			print('  '+color+':', panels[panel].circuits[circuit].getTotalWire()[color])
		print()

	if type(panels[panel].getTotalWire()) != type(0):
		print(panel+' Totals:')
		print(panels[panel].getTotalWire())

		csv_path = csv_dir.joinpath(f'{panel}.csv')

		with open(csv_path, 'w+') as csv_file:
			csv_writer = csv.writer(csv_file)
			csv_writer.writerow( ['color'] + [i for i in WireType.cmils.keys()] )
			A = panels[panel].colors[1]
			B = panels[panel].colors[2]
			C = panels[panel].colors[0]
			N = panels[panel].colors[3]
			G = panels[panel].colors[4]
			ordered_color_list = [A,B,C,N,G]
			n = 0
			for row in panels[panel].getTotalWire():
				color = ordered_color_list[n]
				csv_writer.writerow( [color] + [i for i in row] )
				n += 1