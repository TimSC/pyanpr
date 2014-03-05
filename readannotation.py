import os

def ReadPlateAnnotation(fina):
	fi = open(fina, "rt")
	records = []
	record = []
	subrec = {}

	for li in fi.readlines():
		lis = li.strip()

		if lis=="########## NEW FILE ##########":
			if len(subrec) > 0:
				record.append(subrec)
			if len(record) > 0:
				records.append(record)
				#print record
			record = []
			subrec = {}
			continue

		if lis=="":
			if len(subrec) > 0:
				record.append(subrec)			
			subrec = {}
			continue

		splitPos = lis.find(":")
		if splitPos < 0: continue

		key = lis[:splitPos]
		val = lis[splitPos+2:]
		if key == "object":
			subrec[key] = int(val)
			continue

		if key == "bbox":
			subrec[key] = map(float, val.split(","))
			continue

		subrec[key] = val

	return records

def GetActualImageFileName(baseName, possibleLocations):
	im = None
	finaSplit = os.path.split(baseName)
	actualName = None

	for imgPath in possibleLocations:
		if imgPath is None: continue
		altFina = imgPath+"/"+finaSplit[1]
		if os.path.isfile(altFina):
			actualName = altFina
		if actualName is not None:
			break

	if os.path.isfile(baseName) and actualName is None:
		actualName = baseName

	return actualName

if __name__=="__main__":
	plates = ReadPlateAnnotation("plates.annotation")
	chHist = {}

	#Determine frequency of each character
	for photo in plates:
		for data in photo[1:]:
			for ch in data['reg']:
				if ch == " ": continue
				if ch not in chHist:
					chHist[ch] = 0
				chHist[ch] += 1

	chHistKeys = chHist.keys()
	chHistKeys.sort()
	for ch in chHistKeys:
		print ch, chHist[ch]



