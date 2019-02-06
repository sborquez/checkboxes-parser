"""
Set de coordenadas:
(0,0): esquina superior izquierda
(x,y): 


"""
import cv2
import numpy as np 

class Point():
	"""

	"""
	def __init__(self, x, y, size):
		self.x = x
		self.y = y
		self.size = size #size
		self.center = (50,50)

	def render(self, image):
		#return cv2.circle(image, (self.x, self.y), self.size, (0,0,0), cv2.CV_FILLED, lineType=8, shift=0)
		return cv2.circle(image, (self.x,self.y), self.size, (0,0,0), -1)

	def get_tuple(self):
		return (self.x, self.y)

class Line():
	"""
	"""
	def __init__(self, p1, p2, size, color = (0,0,0)):
		self.p1 = p1
		self.p2 = p2
		self.size = size
		self.color = color

	def get_tangent(self, unit="rad"):
		x1 = self.p1.x
		y1 = self.p1.y
		x2 = self.p2.x
		y2 = self.p2.y
		if (x2-x1) == 0:
			return np.nan
		res = (y2-y1)/(x2-x1)
		res = np.arctan(res)
		if unit == "rad":
			return res
		else:
			esc = 180.0/3.1415
			return res*esc
		

	def get_euclid_length(self):
		x1 = self.p1.x
		y1 = self.p1.y
		x2 = self.p2.x
		y2 = self.p2.y
		return np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))

	def render(self, image):
		x1 = self.p1.x
		y1 = self.p1.y
		x2 = self.p2.x
		y2 = self.p2.y
		aux = cv2.line(image,(x1,y1),(x2,y2),self.color,5)
		return aux

class MOCR_BBox():
	"""
	Bounding box estilo microsoft ocr!
	"""
	def __init__(self, points, size):
		"""
		points es un array de 8 valores: [p1x, p1y, ..., .p4x, p4y]
		p1: punto superior - izquierdo
		p2: punto superior - derecho
		p3: punto inferior - derecho 
		p4: punto inferior - izquierdo. 
		NOTAR: por la orientacion de la palabra, puede que la posicion real de los puntos no tenga 
		la relevancia semantica dada en este help
		"""
		self.p1 = Point(points[0], points[1], size)
		self.p2 = Point(points[2], points[3], size)
		self.p3 = Point(points[4], points[5], size)
		self.p4 = Point(points[6], points[7], size)

		self.l1 = Line(self.p1,self.p2, size, (0,255,0))
		self.l2 = Line(self.p2,self.p3, size, (0,0,0))
		self.l3 = Line(self.p3,self.p4, size, (255,0,0))
		self.l4 = Line(self.p4,self.p1, size, (0,0,255))

		self.size = size

	def render(self, image):
		aux = self.l1.render(image)
		aux = self.l2.render(aux)
		aux = self.l3.render(aux)
		aux = self.l4.render(aux)
		return aux

	def get_tangents(self):
		m1 = self.l1.get_tangent(unit="deg")
		m2 = self.l2.get_tangent(unit="deg")
		m3 = self.l3.get_tangent(unit="deg")
		m4 = self.l4.get_tangent(unit="deg")

		return m1,m2,m3,m4

	def get_orientation(self):
		m1,m2,m3,m4 = self.get_tangents()
		m1 = np.abs(m1)
		m2 = np.abs(m2)
		m3 = np.abs(m3)
		m4 = np.abs(m4)

		if m1 > 88:
			if self.p1.y < self.p2.y:
				return "landscape  tilted right", 90
			else:
				return "landscape tilted left", -90
		elif m1 < 2:
			if self.p1.x < self.p2.x:
				return "portrait"
			else:
				return "portrait upside down", 180
		return "unknown"

	def get_centroid(self):
		"""
		get_centroid calcula el centroide de un bounding box.
		"""
		x1,y1 = self.p1.get_tuple()
		x2,y2 = self.p2.get_tuple()
		x3,y3 = self.p3.get_tuple()
		x4,y4 = self.p4.get_tuple()
		xc = (x1 + x2 + x3 + x4 )/4
		yc = (y1 + y2 + y3 + y4 )/4
		return (xc, yc)
		
		"""
		if m1 is None:
			if m3 is None:
		 		return "unknown m1 & m3 None"
			else:
		 		value = m3
		else:
			if m3 is None:
				value = m1
			else:
				value = (m1 + m3)*0.5
		if value >= 0 and value <= 5:
			return "portrait"
		elif value <= -360:
			return "landscape tilted right"
		else: 
			return "unknown value: {}".format(value)
		"""

class Rectangle():
	"""
	"""
	def __init__(self, center_x, center_y, width, height):
		self.width = width
		self.height = height
		self.center_x = center_x
		self.center_y = center_y
		self.upper_left = Point(self.center_y-int(self.height/2.0), self.center_x-int(self.width/2.0))
		self.bottom_right = Point(self.center_y+int(self.height/2.0), self.center_x+int(self.width/2.0))
		self.selected = False
		self.color = (0,0,0)
		print("[INFO] creado rectangulo: {}, {}".format(self.upper_left.get_tuple(), self.bottom_right.get_tuple()))

	def check_point_is_inside(self, point):
		"""
		point[0] = y
		point[1] = x
		"""
		p1 = self.upper_left.get_tuple()
		p2 = self.bottom_right.get_tuple()
		print("p1: {}, p2: {}, point: {}".format(p1,p2,point))
		if not (point[0] >= p1[0] and point[0] <= p2[0]):
			return False
		if not (point[1] >= p1[1] and point[1] <= p2[1]):
			return False
		return True

	def shift(self, px, py):
		self.center_x += px
		self.center_y += py

	def render(self, image):
		self.upper_left = Point(self.center_y-int(self.height/2.0), self.center_x-int(self.width/2.0))
		self.bottom_right = Point(self.center_y+int(self.height/2.0), self.center_x+int(self.width/2.0))
		p1 = self.upper_left.get_tuple()
		p2 = self.bottom_right.get_tuple()
		aux = cv2.rectangle(image, p1, p2, self.color, 1)
		aux = self.upper_left.render(aux)
		aux = self.bottom_right.render(aux)
		return aux

	def toggle_select(self):
		self.selected = not self.selected
		if self.selected:
			self.color = (0,255,0)
		else:
			self.color = (0,0,0)
