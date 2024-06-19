import json, pyrevit
from pyrevit import revit, forms

doc = pyrevit._DocsGetter().doc
selection = revit.get_selection()


def getLocation(elem):
	location = tuple(float(n) for n in str(elem.Location.Point)[1:][:-1].split(sep=','))
	return location


def getPanelVoltage(panel):
	if str(panel.LookupParameter('Electrical Data').AsString()).__contains__('480'):
		voltage = 480
	elif str(panel.LookupParameter('Electrical Data').AsString()).__contains__('208'):
		voltage = 208
	else:
		voltage = 0
	return voltage


def getCircNums(circuit):
	circ_nums = tuple([ int(num) for num in circuit.CircuitNumber.split(sep=',') ])
	return circ_nums


def getRunElev(circuit):
	devices = []
	for device in circuit.Elements:
		devices.append(device)
	if getLocation(devices[0])[2] <= 10:
		run_elev = -1
	else:
		run_elev = getLocation(devices[0])[2]
	return run_elev


def getTripAmps(circuit):
	trip_amps = int(circuit.LookupParameter('Rating').AsValueString()[:-2])
	return trip_amps

panel_data = {}

try:
	assert selection != None
except:
	print('ERROR: Please select one or more Electrical Panels.')
else:
	for panel in selection:
		panel_data[panel.Name] = {}
		panel_data[panel.Name]['voltage'] = getPanelVoltage(panel)
		panel_data[panel.Name]['location'] = getLocation(panel)
		panel_data[panel.Name]['circuits'] = {}

		panel_elec_systems = panel.MEPModel.GetAssignedElectricalSystems()
		if panel_elec_systems.Count == 0:
			continue
		else:
			for circuit in panel_elec_systems:
				if circuit.Elements == None:
					continue
				elif circuit.Elements.IsEmpty:
					continue
				else:
					circuit_name = str(getCircNums(circuit))
					panel_data[panel.Name]['circuits'][circuit_name] = {}
					panel_data[panel.Name]['circuits'][circuit_name]['panel'] = panel.Name
					panel_data[panel.Name]['circuits'][circuit_name]['circ_nums'] = getCircNums(circuit)
					panel_data[panel.Name]['circuits'][circuit_name]['trip_amps'] = getTripAmps(circuit)
					panel_data[panel.Name]['circuits'][circuit_name]['run_elev'] = getRunElev(circuit)
					panel_data[panel.Name]['circuits'][circuit_name]['devices'] = {}
					
					for device in circuit.Elements:
						panel_data[panel.Name]['circuits'][circuit_name]['devices'][int(device.Id.ToString())] = getLocation(device)

jsonpath = forms.save_file(file_ext="json",title="Save Panel Data as JSON")

with open(jsonpath,'w') as jsonfile:
	json.dump(panel_data,jsonfile)