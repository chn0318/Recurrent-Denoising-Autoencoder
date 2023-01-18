# RenderDoc Python console, powered by python 3.6.4.
# The 'pyrenderdoc' object is the current CaptureContext instance.
# The 'renderdoc' and 'qrenderdoc' modules are available.
# Documentation is available: https://renderdoc.org/docs/python_api/index.html

filepath = 'E:/test/draft2.rdc'
rd = renderdoc
#rd = renderdoc
#????capture
pyrenderdoc.LoadCapture(filepath, renderdoc.ReplayOptions(), filepath, False, True)

def Callback(controller):
	for d in controller.GetResources():
		print(d.name)
		if d.name=='GBufferA':
			for textures in controller.GetTextures():
				if textures.resourceId==d.resourceId:
					print('nb')
					texsave = rd.TextureSave()
					texsave.resourceId = textures.resourceId
					filename = 'E:/test/'+str(int(texsave.resourceId))
					texsave.alpha = rd.AlphaMapping.BlendToCheckerboard

					# Most formats can only display a single image per file, so we select the
					# first mip and first slice
					texsave.mip = 0
					texsave.slice.sliceIndex = 0
					texsave.destType = rd.FileType.JPG
					controller.SaveTexture(texsave, filename + ".jpg")
					texsave.alpha = rd.AlphaMapping.Preserve
					texsave.destType = rd.FileType.PNG
					controller.SaveTexture(texsave, filename + ".png")
					#controller.GetTextureData(d.resourceId)
			break
def iterAction(d, indent = ''):
	# Print this action
	print('%s%d: %s' % (indent, d.eventId, d.GetName(controller.GetStructuredFile())))

	# Iterate over the action's children
	for d in d.children:
		iterAction(d, indent + '    ')

	


pyrenderdoc.Replay().BlockInvoke(Callback)

