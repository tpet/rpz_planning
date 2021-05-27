import bpy
import os
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:]

# Removing default cube, camera and light,
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Importing and exporting the scene
for input_file in os.listdir(argv[0]):
	input_path = os.path.join(argv[0], input_file)
	print('Importing', input_path)
	bpy.ops.wm.collada_import(filepath=input_path)
bpy.ops.wm.collada_export(filepath=argv[1])

print('-' * 30)
print('  Convertion is done !')
print('-' * 30)
