import sys
import numpy as np
import copy

def main():
	filename = sys.argv[1]
	filehead = filename.split('.par')[0]
	print("\nRUNNING PROGRAM \"passivate_tetrahedral.py\" ON FILE: {}".format(filename))
	try:
		units = sys.argv[2]
	except IndexError:
		units = 'angstrom'
		print('Default units of angstrom are assumed. Change to atomic by sys.argv[2] = atomic')
	try:
		ltype = sys.argv[2]
	except IndexError:
		print('No ligand type specified! Default (Cd -> P1; Se -> P2) will be used.')
		ltype = '\t\t'
		
	atoms = open_file(filename, units)
	
	
	print("\nGetting neighbors...".upper())
	full_neighbors = get_neighbors(atoms)
	
	print("\nRemoving dangling atoms...".upper())
	new_atoms, full_neighbors = rem_dangling_atoms(atoms, full_neighbors)
	
	print("\nFlagging extra bonds...".upper())
	#flag_extra_bonds(new_atoms, full_neighbors)
	
	print("\nWRITING removedDanglingAtoms.xyz...")
	write_output('removedDanglingAtoms', new_atoms, 'xyz')
	
	print('\nPassivating atoms...'.upper())
	natoms = len(new_atoms)
	print('nAtoms = ', natoms)
	atoms_cpy = copy.deepcopy(new_atoms)
	passivated_atoms = [0.0 for i in range(natoms)]
	for i in range(natoms):
		neigh_list = full_neighbors[i]
		
		if neigh_list[1] == 2:
			new_atoms = [atoms_cpy[i], atoms_cpy[neigh_list[0][1]], atoms_cpy[neigh_list[0][2]]]
			tmp_atoms = passivate(new_atoms, 2)
			
			# The neighbor list will slot all Cd/Se atoms back in place so indexing the crystal isn't impossible 
			for j, idx in enumerate(neigh_list[0]):
				passivated_atoms[idx] = tmp_atoms[j]
				
			# the new ligands will be appended
			passivated_atoms.append(tmp_atoms[3])
			passivated_atoms.append(tmp_atoms[4])
		
		elif neigh_list[1] == 3:
			new_atoms = [atoms_cpy[i], atoms_cpy[neigh_list[0][1]], atoms_cpy[neigh_list[0][2]], atoms_cpy[neigh_list[0][3]]]
			tmp_atoms = passivate(new_atoms, 3)
			
			for j, idx in enumerate(neigh_list[0]):
				passivated_atoms[idx] = tmp_atoms[j]
			try:	
				passivated_atoms.append(tmp_atoms[4])
			except IndexError:
				print('i: ', i, '\nnew_atoms: ', new_atoms, '\ntmp_atoms: ', tmp_atoms, '\nneigh_list: ', neigh_list)
				exit()
			
		else:
			for idx in neigh_list[0]:
				passivated_atoms[idx] = atoms_cpy[idx]
				
	print('Writing output files...'.upper())
	write_output(filehead+"_conf", passivated_atoms, 'par')
	write_output(filehead+"_passivated", passivated_atoms, 'H')
	print('\nProgram done!\n'.upper())
	


def open_file(file, units):
	if units == 'angstrom':
		unit_conv = 1.0
	if units == 'atomic':
		unit_conv = 0.529177
		
	with open(file, 'r') as f:
		conf = f.readlines()
		f.close()
	
	n_atoms = conf[0]
	conf = conf[1:]

	labels = []
	coords = []
	for i, line in enumerate(conf):
		line = line.split()
		labels.append(line[0])
		coords.append([float(line[1])*unit_conv, float(line[2])*unit_conv, float(line[3])*unit_conv])

	atoms = []
	for i in range(len(labels)):
		atoms.append( [labels[i]]+coords[i] )
	
	return atoms

def dis(u,v):
	return np.sqrt((v[0]-u[0])**2 + (v[1]-u[1])**2 + (v[2]-u[2])**2)

def write_output(filename, coords, opt='xyz'):
	if opt == 'xyz':
		units = 1.0
		filename += '.xyz'
		fmt = '\n\n'
	if opt == 'par':
		units = 1/0.529177 
		filename += '.par'
		fmt = '\n'
	if opt == 'H':
		units = 1.0
		filename += '_H.xyz'
		fmt = '\n\n'
	
	file = open(filename, 'w')
	
	n_atoms = str(len(coords))
	file.write(n_atoms+fmt)
	
	for i, coord in enumerate(coords):
		if opt == 'H':
			if coord[0] == 'P1' or coord[0] == 'P2':
				coord[0] = 'H'
		
		file.write('{name} {x} {y} {z}\n'.format(name=coord[0],x=coord[1]*units,y=coord[2]*units,z=coord[3]*units))
		
	file.close()

def ctr_atoms(atom_list, idx):
	xshift = atom_list[idx][1]
	yshift = atom_list[idx][2]
	zshift = atom_list[idx][3]
	for i in range(len(atom_list)):
		atom_list[i][1] -= xshift
		atom_list[i][2] -= yshift
		atom_list[i][3] -= zshift
	return atom_list, (xshift, yshift, zshift)

def angle(ctr, atom1, atom2):
	a = np.asarray([atom1[i] - ctr[i] for i in range(3)])
	b = np.asarray([atom2[i] - ctr[i] for i in range(3)])
	arg = a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))
	return np.arccos(arg)		

def get_neighbors(atoms):
	full_neighbors = []
	for i, atom1 in enumerate(atoms):
		neighbor_list = []
		neighbor_count = 0
		for j, atom2 in enumerate(atoms):
			if i == j: # Use this to make the first index of neighbor list the identity of the center atom.
				neighbor_list.insert(0,j)
				continue
			if dis(atom1[1:], atom2[1:]) < rad_cutoff(atom1[0], atom2[0]):
				neighbor_list.append(j)
				neighbor_count += 1

		full_neighbors.append((neighbor_list, neighbor_count))

	return full_neighbors
	
def rad_cutoff(atom1, atom2):
	if (atom1 == 'Cd' and atom2 == 'Se') or (atom1 == 'Se' and atom2 == 'Cd'):
		return 2.63 * 1.15
	else:
		return 0.0

def skew_symmetric(vec):
	
	mat = [\
	[0, -vec[2], vec[1]],
	[vec[2], 0, -vec[0]],
	[-vec[1], vec[0], 0]]
	
	return np.asarray(mat)

def rot_mat1(ctr, atom1, atom2):
	'''https://math.stackexchange.com/questions/180418/
	calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d'''
	
	a = np.asarray(atom1) - np.asarray(ctr)
	b = np.asarray(atom2) - np.asarray(ctr)
	
	a /= np.linalg.norm(a)
	b /= np.linalg.norm(b)
	
	theta = angle(ctr, a, b)
	
	std_point1 = [np.sin(theta/2), np.cos(theta/2), 0.0]
	
	v = np.cross(a,std_point1)
	c = np.dot(a,std_point1) # to get angle between old and new vectors
	
	v_x = skew_symmetric(v)
	
	R_1 = np.eye(3) + v_x + np.matmul(v_x,v_x) * (1/(1+c))
	return R_1
	
def rot_mat2(ctr, atom1, atom2):
	# https://math.stackexchange.com/questions/142821/matrix-for-rotation-around-a-vector
	
	# v is the bond vector we have aligned to Se-Cd1 in the "standard orientation." 
	# It defines the axis about which we will rotate. 
	l1, l2 = np.linalg.norm(atom1), np.linalg.norm(atom2)  # bond length of v
	v = np.asarray(atom1)/l1 # the vector we will rotate around.
	
	# u is the position of the randomly oriented second atom. 
	# We will rotate it so it aligns with Se-Cd2 in the standard orientation
	u = np.asarray(atom2)/l2
	
	# Errors can arise if the second atom is already in the plane because the arccos function
	# is numerically unstable near 0 and pi. If u_z is close to zero, we will rotate it by 30 degrees and proceed
	if np.isclose(u[2],0.0):
		#print('z-component of atom2 is close to zero: ', u, ' Rotating by 30 deg.\n')
		theta = 30*(np.pi/180)
		v_x = skew_symmetric(v)
		R_2 = np.eye(3) + np.sin(theta)*v_x + (2 * np.sin(theta/2)**2)* np.matmul(v_x,v_x)
		u = np.matmul(R_2, u)
		#print('new atom2: ', u*l2)
	
	# Phi is the acute angle that u makes with the altitude of v
	phi = np.pi - angle(ctr, v, u)
	
	a = l2*np.sin(phi) # radius of cone swept out by randomly oriented vector. Point Se-Cd2 is on this cone.
	
	pi_minus_phi = np.pi - phi # angle needed to calculate the standard point 2 (Se-Cd2)
	std_point2 = [-l2*np.sin(pi_minus_phi/2), l2*np.cos(pi_minus_phi/2), 0.0] # Se-Cd2
	
	c = dis(u*l2,std_point2) # For law of cosines (see Tetrahedral Passivation)
	
	theta = np.arccos(1 - (c**2)/(2*a**2)) # Angle of rotation to align with Se-Cd2. See "Tetrahedral Passivation" document

	if u[2] > 0:
		theta = 2*np.pi-theta
	
	v_x = skew_symmetric(v)
	
	R_2 = np.eye(3) + np.sin(theta)*v_x + (2 * np.sin(theta/2)**2)* np.matmul(v_x,v_x)
	#R_2 = np.cos(theta)*np.eye(3) + np.sin(theta)*v_x + (1 - np.cos(theta))* np.outer(v,v)
	return R_2

def rot_atoms(atoms, R):
	new_atoms = []
	for atom in atoms:
		#print('atom in rot_atoms: ', atom)
		new_atoms.append([atom[0]] + np.matmul(R, atom[1:]).tolist())
	return new_atoms

def passivate(atom_list, nbonds):
	#Passivate atoms with coordination vacancies
	if nbonds != 2 and nbonds != 3:
		print("Error: trying to passivate ligand with nbonds != 2 or 3")
		exit()
	
	new_atoms = copy.deepcopy(atom_list)
	new_atoms, translation = ctr_atoms(new_atoms, 0)
	
	# Grab center atom and its two neighbors
	ctr = new_atoms[0]
	atom1 = new_atoms[1][1:]
	atom2 = new_atoms[2][1:]
	
	R1 = rot_mat1(ctr[1:], atom1, atom2) # Construct rotation matrix to align with Se-Cd1
	tmp_atoms = rot_atoms(new_atoms, R1) # Rotate to align with Se-Cd1
	
	tmp_atom1 = tmp_atoms[1][1:] # just for readability of the next function call
	tmp_atom2 = tmp_atoms[2][1:]
	
	R2 = rot_mat2(ctr[1:], tmp_atom1, tmp_atom2) # Construct rotation matrix to align with Se-Cd1
	tmp_atoms = rot_atoms(tmp_atoms, R2) # rotate to align with Se-Cd1
	
	# Passivate the configuration
	tmp_atom1 = tmp_atoms[1][1:] # just for readability of the next function call
	tmp_atom2 = tmp_atoms[2][1:]
	
	if nbonds == 2:
		tmp_atoms = add_2_ligands(tmp_atoms, ctr, tmp_atom1, tmp_atom2)
	elif nbonds == 3:
		tmp_atom3 = tmp_atoms[3][1:]
		tmp_atoms = add_1_ligand(tmp_atoms, ctr, tmp_atom1, tmp_atom2, tmp_atom3)
	
	# Invert the transformations
	tmp_atoms = rot_atoms(tmp_atoms, np.transpose(R2))
	tmp_atoms = rot_atoms(tmp_atoms, np.transpose(R1))
	
	for i in range(len(tmp_atoms)):
		tmp_atoms[i][1] += translation[0]
		tmp_atoms[i][2] += translation[1]
		tmp_atoms[i][3] += translation[2]
	
	return tmp_atoms
			
def add_2_ligands(atoms, ctr, atom1, atom2):
	# Contract passivation bond length depending on whether atom is cation or anion
	if ctr[0] == 'Cd':
		scalef = 0.55
		ltype = 'P1'
	elif ctr[0] == 'Se':
		scalef = 0.25
		ltype = 'P2'
	
	l = dis(ctr[1:], atom1) * scalef
	theta = angle(ctr[1:], atom1, atom2)
	lig1 = [0, -l * np.cos(theta/2), l * np.sin(theta/2)]
	lig2 = [0, -l * np.cos(theta/2), -l * np.sin(theta/2)]
	atoms.append([ltype] + lig1)
	atoms.append([ltype] + lig2)
	return atoms

def add_1_ligand(atoms, ctr, atom1, atom2, atom3):
	# Contract passivation bond length depending on whether atom is cation or anion
	if ctr[0] == 'Cd':
		scalef = 0.55
		ltype = 'P1'
	elif ctr[0] == 'Se':
		scalef = 0.30
		ltype = 'P2'

	l = dis(ctr[1:], atom1) * scalef
	theta = angle(ctr[1:], atom1, atom2)
	lig1 = [0, -l * np.cos(theta/2), l * np.sin(theta/2)]
	lig2 = [0, -l * np.cos(theta/2), -l * np.sin(theta/2)]
	# Check whether or not atom3 is in the lig1 or lig2 position.
	# If atom3 is above the xy plane, then we want to passivate at lig2
	if atom3[2] > 0: 
		atoms.append([ltype] + lig2)
	elif atom3[2] < 0: 
		atoms.append([ltype] + lig1)
	return atoms

def rem_dangling_atoms(atoms, neigh_list, min_bonds = 2):
	neighs = copy.deepcopy(neigh_list)
	new_atoms = copy.deepcopy(atoms)
	n_tmp = len(atoms)
	n_iter = 0
	n_new = 0
	while n_new != n_tmp:
		n_iter += 1
		tmp_atoms = copy.deepcopy(new_atoms)
		if n_iter > 1: n_tmp = n_new
		print("\n\nIteration: %d" %n_iter)
		print("Starting number of atoms: %d" %n_tmp)
		n_new = len(new_atoms)
		
		for idx, neigh_count in enumerate(neighs): # Neighbor list has [neighbors, neigh_count]
			if neigh_count[1] < min_bonds: # If the neighbor count is more than the min, let the atom into the new_atoms list
				tmp_atoms[neigh_count[0][0]] = 0.0
				n_new -= 1
		
		if n_new != n_tmp:
			print('{} atoms removed'.format(n_tmp-n_new))
			tmp_atoms = np.asarray(tmp_atoms, dtype='object')
			new_atoms = list(tmp_atoms[tmp_atoms[:] != 0.0])
			print("\n\tRecalculating neighbor list for next cycle...")
			neighs = get_neighbors(new_atoms)
		else:
			print('Done removing dangling atoms!')
			new_atoms = tmp_atoms
		#write_output("iter_{}".format(n_iter), new_atoms, 'xyz')
	return new_atoms, neighs

def flag_extra_bonds(atoms, neigh_list, max_bonds = 4):
	for idx, neigh_count in enumerate(neigh_list): # Neighbor list has [neighbors, neigh_count]
		if neigh_count[1] > max_bonds: # If the neighbor count is more than the min, let the atom into the new_atoms list
			print('FLAG: Atom {}{}'.format(atoms[idx][0],idx), ' has {} neighbors!'.format(neigh_count[1]))


main()
