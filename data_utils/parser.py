def list_to_string(lst):
	return ', '.join(lst)

def consolidate(lst: list):
	return

def parse_catalogue(raw_catalogue):
	cleaned_catalogue = {}
	for item_iter in raw_catalogue:
		if (item_iter['isSearchable']):
			item_dict = {}

			if item_iter['taxCode'] not in ['GC', 'GW']:

				# Style
				item_dict['style'] = item_iter['styleDescription'][0]['value']

				# Colors
				# Price
				try:
					colors = []
					rprice = []
					cprice = []
					ptype = []
					for i in item_iter['customerChoice']:
						color = i['searchColor']['id']
						colors.append(color)
						reg_price = i['price']['regularPrice']
						cur_price = i['price']['currentPrice']
						price_type = i['price']['priceType']
						rprice.append(reg_price)
						cprice.append(cur_price)
						ptype.append(price_type)

					item_dict["colors"] = list_to_string(colors)
					item_dict["regular_price"] = list(set(rprice))
					item_dict["current_price"] = list(set(cprice))
					item_dict["price_type"] = list(set(ptype))
				except KeyError:
					item_dict["colors"] = None
					item_dict["regular_price"] = []
					item_dict["current_price"] = []
					item_dict["price_type"] = []

				# Fabric
				try:
					fabrics = []
					for i in item_iter['fabricCopy']['bullets']:
						if len(i['bulletContent']) > 0:
							fabric = i['bulletContent'][0]['value'] + " "
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
							fit = i['bulletContent'][0]['value'] + " "
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
					overview = i['bulletContent'][0]['value'] + " "
					overviews.append(overview)
			if len(overviews) > 0:
				item_dict["overviews"] = list_to_string(overviews)
			else:
				item_dict["overviews"] = None

			# URL
			try:
				if item_iter["customerChoice"]:
					item_dict["url"] = item_iter["customerChoice"][0]['fullProductPageURL']
				else:
					item_dict["url"] = None
			except KeyError:
				item_dict["url"] = None

			try:
				item_dict["avg_rating"] = item_iter['rating']['averageRating']
			except KeyError:
				item_dict["avg_rating"] = None

			style_num = item_iter['universalStyleNumber']
			cleaned_catalogue[style_num] = item_dict

	return cleaned_catalogue