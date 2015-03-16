import os

import matplotlib
import matplotlib.pyplot as plt
import muscip.connectome as mcon
import nipype.interfaces.diffusion_toolkit as dtk
import nipype.interfaces.fsl as fsl
import numpy as np

def _to_dict(**kw): return kw

def run(sandbox_dir, data, t1, roi, mni, rewrite = True, bvecs = None, bvals = None, start_from = None, run_until = None, eddy = None, brain = None, t1_brain = None, fa = None, fa_erode = None, fa_thresh = None, diff_roi = None, roi_mask = None, tensor = None, stream_lines = None):
	"""generate the connectome
	if bvecs or bvals are not provided, uses bvecs and bvals files in directory of 'data'
	start_from specifies which step to start calculating from, valid values in step order:
		['eddy', 'brain', 't1_brain', 'fa', 'diff_roi', 'roi_mask', 'tensor', 'stream_lines', 'connectome']
	run_until specifies which step to run until (includes)
	if specifying a start_from, also specify the inputs that the step requires
		for example, if start_from = 'brain', then the input 'eddy' must be set
		the most robust method to determine what inputs are required for a step is to view the source code
	if a given step is set, then the step will not be recalculated:
		ie: if eddy input set to 'path_to_eddy', then eddy will not be recalculated
	"""
	if os.path.isdir(sandbox_dir) and rewrite:
		import shutil
		shutil.rmtree(sandbox_dir)

	if not os.path.isdir(sandbox_dir):
		os.mkdir(sandbox_dir)

	input_dir = os.path.dirname(data)
	if not bvecs:
		bvecs = os.path.join(input_dir, 'bvecs')
	if not bvals:
		bvals = os.path.join(input_dir, 'bvals')

	os.chdir(sandbox_dir)

	steps_dict = _to_dict(eddy = eddy, brain = brain, t1_brain = t1_brain, fa = fa, diff_roi = diff_roi, roi_mask = roi_mask, tensor = tensor, stream_lines = stream_lines, connectome = None)
	steps = ['eddy', 'brain', 't1_brain', 'fa', 'diff_roi', 'roi_mask', 'tensor', 'stream_lines', 'connectome']

	def within(step):
		return (not start_from or steps.index(step) >= steps.index(start_from)) and (
				not run_until or steps.index(step) <= steps.index(run_until))

	def calc_if_necessary(step, fn):
		input_val = steps_dict[step]
		if input_val:
			return input_val
		elif within(step):
			return fn()

	eddy = calc_if_necessary('eddy', lambda: motion_correction(data))

	brain = calc_if_necessary('brain', lambda: brain_extraction(eddy))

	t1_brain = calc_if_necessary('t1_brain', lambda: brain_extraction_simple(t1, sandbox_dir))

	fas = calc_if_necessary('fa', 
			lambda: prepare_fa_maps(eddy, brain, bvecs, bvals))
	if fas is not fa:
		fa = fas[0]
		fa_erode = fas[1]
		fa_thresh = fas[2]

	diff_roi = calc_if_necessary('diff_roi', 
			lambda: register_rois(fa_erode, t1_brain, mni, roi))

	roi_mask = calc_if_necessary('roi_mask', 
			lambda: remove_high_fa_from_roi(diff_roi, fa_thresh))

	tensor = calc_if_necessary('tensor', 
			lambda: generate_tensor(eddy, bvecs, bvals))

	stream_lines = calc_if_necessary('stream_lines',
			lambda: generate_stream_lines(eddy, fa, tensor))

	calc_if_necessary('connectome', lambda: compile_connectome(stream_lines, roi_mask))


def compile_connectome(fibers, roi_mask):
	C = mcon.TNDtkConnectome(fibers = fibers,
			min_fiber_length = 5.0,
			max_fiber_length = 500.0,
			roi_image = roi_mask)
	C.generate_network()
	C.write('ctome')

def show_ctome(ctome_path, log = True):
	C = mcon.read(ctome_path)
	fiber_counts = C.matrix_for_key('fiber_count')
	if log:
		fiber_counts = np.log(1 + fiber_counts)

	fig = plt.figure('Connectome Matrix')
	plt.imshow(fiber_counts)
	plt.colorbar()
	fig.tight_layout()
	plt.show()

def generate_stream_lines(data, fa_mask, tensor):

	dti_tracker = dtk.DTITracker()
	dti_tracker.inputs.tensor_file = tensor
	dti_tracker.inputs.input_data_prefix = os.path.realpath('.')
	dti_tracker.inputs.input_type = 'nii'
	dti_tracker.inputs.output_file = 'streamlines.trk'
	dti_tracker.inputs.mask1_file = fa_mask
	dti_tracker.inputs.invert_y = True
	dti_tracker.inputs.args = '.2 1'

	_run(dti_tracker)

	return dti_tracker.inputs.output_file

def generate_tensor(data, bvecs, bvals):
	#dti_recon "/Users/john/Desktop/1138a_processed/data_eddy.nii" "/Users/john/Desktop/" -gm "/var/folders/rq/m3ktkh5s6nxb5n2spysg5fkw0000gn/T/dtk_tmp/matrices/gradient.txt" -b 1000 -b0 auto -iop 1 0 0 0 1 0  -p 3 -sn 1 -ot nii 
	tensor = dtk.DTIRecon()
	tensor.inputs.bvecs = bvecs
	tensor.inputs.bvals = bvals
	tensor.inputs.args = '-iop 1 0 0 0 1 0 -b 1000 -b0 auto -p 3 -sn 1'
	tensor.inputs.DWI = data
	tensor.inputs.output_type = 'nii'
	run_res = tensor.run()
	return run_res.outputs.get()['tensor']

def _run(cmd):
	print(cmd.cmdline)
	cmd.run()

def motion_correction(input_file):
	out_file =  insert_into_filename(os.path.basename(input_file), '_eddy')

	eddyc = fsl.EddyCorrect(in_file=input_file, out_file = out_file, ref_num = 0)

	_run(eddyc)
	return out_file

def brain_extraction(in_file, roi_file = 'b0.nii.gz', t_min = 0, t_size = 1):
	from nipype.interfaces.fsl import ExtractROI
	fslroi = ExtractROI(in_file=in_file, roi_file=roi_file, t_min = t_min, t_size = t_size)
	_run(fslroi)

	bet = fsl.BET()
	bet.inputs.in_file = roi_file
	bet.inputs.frac = .3
	bet.inputs.args = '-n -m'

	out_file = insert_into_filename(in_file, '_brain')
	bet.inputs.out_file = out_file
	_run(bet)
	return insert_into_filename(out_file, '_mask')

def brain_extraction_simple(in_file, output_dir = None):
	bet = fsl.BET()
	bet.inputs.in_file = in_file
	bet.inputs.args = '-m'
	out_file = insert_into_filename(in_file, '_brain')

	if output_dir:
		basename = os.path.basename(out_file)
		out_file = os.path.join(output_dir, basename)

	bet.inputs.out_file = out_file
	_run(bet)
	return bet.inputs.out_file

def prepare_fa_maps(input_file, brain_mask, bvecs = 'bvecs', bvals = 'bvals', fa_thresh = 0.20):
	dti = fsl.DTIFit()
	dti.inputs.dwi = input_file
	dti.inputs.bvecs = bvecs
	dti.inputs.bvals = bvals
	dti.inputs.base_name, tmp = file_ext(os.path.basename(input_file))
	dti.inputs.mask = brain_mask 

	_run(dti)

	fa = insert_into_filename(input_file, '_FA')
	
	erode_input = fa
	erode_output = insert_into_filename(erode_input, '_ero')
	erode = fsl.ErodeImage()
	erode.inputs.in_file = erode_input 
	erode.inputs.out_file = erode_output
	_run(erode)

	thresh_input = erode_output
	thresh_output = insert_into_filename(erode_output, '_thresh')
	thresh = fsl.Threshold()
	thresh.inputs.in_file = thresh_input
	thresh.inputs.out_file = thresh_output
	thresh.inputs.output_datatype = 'char'
	thresh.inputs.thresh = fa_thresh
	_run(thresh)

	return (fa, erode_output, thresh_output)

def register_rois(diff_fa, t1, std, roi):

	std_to_t1 = fsl.FLIRT()
	std_to_t1.inputs.in_file = std
	std_to_t1.inputs.reference = t1
	std_to_t1.inputs.out_matrix_file = 'std_to_t1.mat'
	std_to_t1.inputs.out_file = 'std_to_t1.nii.gz'
	_run(std_to_t1)

	roi_to_t1 = fsl.FLIRT()
	roi_to_t1.inputs.in_file = roi
	roi_to_t1.inputs.reference = t1
	roi_to_t1.inputs.out_matrix_file = 'roi_to_t1.mat'
	roi_to_t1.inputs.out_file = 'roi_to_t1.nii.gz'
	roi_to_t1.inputs.in_matrix_file = std_to_t1.inputs.out_matrix_file
	_run(roi_to_t1)

	fa_sqr = fsl.UnaryMaths()
	fa_sqr.inputs.in_file = diff_fa
	fa_sqr.inputs.out_file = insert_into_filename(diff_fa, '_sqr')
	fa_sqr.inputs.operation = 'sqr'
	_run(fa_sqr)


	t1_to_fa_sqr = fsl.FLIRT()
	t1_to_fa_sqr.inputs.in_file = t1
	t1_to_fa_sqr.inputs.reference = fa_sqr.inputs.out_file
	t1_to_fa_sqr.inputs.out_matrix_file = 't1_to_fa_sqr.mat'
	t1_to_fa_sqr.inputs.out_file = 't1_to_fa_sqr.nii.gz'

	_run(t1_to_fa_sqr)
	

	roi_t1_to_diff = fsl.FLIRT()
	roi_t1_to_diff.inputs.in_file = roi_to_t1.inputs.out_file 
	roi_t1_to_diff.inputs.reference = diff_fa
	roi_t1_to_diff.inputs.in_matrix_file = t1_to_fa_sqr.inputs.out_matrix_file
	roi_t1_to_diff.inputs.out_matrix_file = 'roi_t1_to_diff.mat'
	roi_t1_to_diff.inputs.out_file = 'roi_t1_to_diff.nii.gz'

	_run(roi_t1_to_diff)

	"""
	std_to_diff = fsl.FLIRT()

	std_to_diff.inputs.reference = diff_fa
	std_to_diff.inputs.out_matrix_file = 'std_to_diff.mat'
	std_to_diff.inputs.out_file = 'std_to_diff.nii.gz'
	std_to_diff.inputs.in_file = std

	std_to_diff.inputs.cost = 'corratio'
	std_to_diff.inputs.bins = 256
	std_to_diff.inputs.searchr_x = [45, 45]
	std_to_diff.inputs.searchr_y = [45, 45]
	std_to_diff.inputs.searchr_z = [45, 45]
	std_to_diff.inputs.dof = 12
	std_to_diff.inputs.interp = 'trilinear'

	_run(std_to_diff)

	roi_to_diff = fsl.FLIRT()
	
	roi_to_diff.inputs.in_file = roi
	roi_to_diff.inputs.reference = diff_fa
	roi_to_diff.inputs.out_file = 'roi_to_diff.nii.gz'
	roi_to_diff.inputs.in_matrix_file = std_to_diff.inputs.out_matrix_file
	roi_to_diff.inputs.apply_xfm = True

	roi_to_diff.inputs.interp = 'nearestneighbour'

	_run(roi_to_diff)

	return roi_to_diff.inputs.out_file
	"""

	return roi_t1_to_diff.inputs.out_file

def remove_high_fa_from_roi(roi, fa):
	fa_inv = fsl.UnaryMaths()
	fa_inv.inputs.in_file = fa
	fa_inv.inputs.out_file = insert_into_filename(fa, '_inv')
	fa_inv.inputs.operation = 'binv'

	_run(fa_inv)

	mask_roi = fsl.ApplyMask()
	mask_roi.inputs.in_file = roi
	mask_roi.inputs.mask_file = fa_inv.inputs.out_file
	mask_roi.inputs.out_file = insert_into_filename(roi, '_thr')

	_run(mask_roi)

	return mask_roi.inputs.out_file

def insert_into_filename(filename, insert):
	(name, ext) = file_ext(filename)
	return name + insert + ext

def file_ext(filename, acc = ''):
	(rest, ext) = os.path.splitext(filename)
	full_ext = ext + acc if acc else ext
	if ext:
		return file_ext(rest, full_ext)
	else:
		return (rest, full_ext)
