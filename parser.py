def list_to_string(lst):
	return ', '.join(lst)


def parse_catalogue(raw_catalogue):
	cleaned_catalogue = {}
	for item_iter in raw_catalogue:
		if (item_iter['isSearchable']):
			item_dict = {}

			if item_iter['taxCode'] not in ['GC', 'GW']:

				# Style
				item_dict['style'] = item_iter['styleDescription'][0]['value']

				# Colors
				try:
					colors = []
					for i in item_iter['customerChoice']:
						color = i['searchColor']['id']
						colors.append(color)
					item_dict["colors"] = list_to_string(colors)
				except KeyError:
					item_dict["colors"] = None

				# Fabric
				try:
					fabrics = []
					for i in item_iter['fabricCopy']['bullets']:
						if len(i['bulletContent']) > 0:
							fabric = i['bulletContent'][0]['value'] + "."
							fabrics.append(fabric)
					if len(fabrics) > 0:
						item_dict["fabrics"] = list_to_string(fabrics)
					else:
						item_dict["fabrics"] = None
				except KeyError:
					item_dict["fabrics"] = None

				# Fits
				try:
					fits = []
					for i in item_iter['fitAndSizingCopyBullets']:
						if len(i['bulletContent']) > 0:
							fit = i['bulletContent'][0]['value'] + "."
							fits.append(fit)
					if len(fits) > 0:
						item_dict["fits"] = list_to_string(fits)
					else:
						item_dict["fits"] = None
				except KeyError:
					item_dict["fits"] = None

				# Tags
				tags = []
				for i in item_iter['productTags']:
					tag = i['tags'][0]['tagName'][0]['value']
					tags.append(tag)
				item_dict["tags"] = list_to_string(tags)

				# Hierarchy
				hierarchys = []
				for level in ["divisionDescription", "departmentDescription", "classDescription", "subclassDescription"]:
					hierarchy = item_iter['universalMerchandiseHierarchy'][level]
					hierarchys.append(hierarchy)
				item_dict["hierarchys"] = list_to_string(hierarchys)

			# Overview
			overviews = []
			for i in item_iter['overviewCopy']['bullets']:
				if len(i['bulletContent']) > 0:
					overview = i['bulletContent'][0]['value'] + "."
					overviews.append(overview)
			if len(overviews) > 0:
				item_dict["overviews"] = list_to_string(overviews)
			else:
				item_dict["overviews"] = None

			try:
				item_dict["avg_rating"] = item_iter['rating']['averageRating']
			except KeyError:
				item_dict["avg_rating"] = None

			style_num = item_iter['universalStyleNumber']
			cleaned_catalogue[style_num] = item_dict

	return cleaned_catalogue