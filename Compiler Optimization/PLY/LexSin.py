import ply.lex as lex
import ply.yacc as yacc
import sys
import codecs
import os

loc = 'C:/Proyecto/Pruebas/'

tc = 0
pc = 0 
D = 0
i = 0

varMap = []
stackOp = []
stackJmp = []
stackPrg = []
stackTemp = []
stackMod = []

#Variable de tabla de simbolos
class SymTab:
	def __init__(self, varID, varTP, varVal):
		self.varID = varID
		self.varTP = varTP
		self.varVal = varVal
	def out(self):
		print("\tID: " + str(self.varID) + "\tTYPE: " + self.varTP+ "\tVALUE: " + str(self.varVal))

class Cuad:
	def __init__(self, op, v1, v2, d):
		self.op = op
		self.v1 = v1
		self.v2 = v2
		self.d = d
	def out(self):
		print(self.op + " " + self.v1 + " " + self.v2 + " " + self.d)
	def exe(self, J):
		#-------------------------#
		# Variable ID value fetch #
		#-------------------------#
		if (self.v1 != "" and self.v1[0] != "\""):
			if (self.v1[0] == "_"):
				x = searchVar(self.v1)
			elif (self.v1[0] == "T"):
				x = stackTemp.pop()
			else: 
				if (self.v1.find(".") != -1):
					x = SymTab("X","float",float(self.v1))
				else: 
					x = SymTab("X","int",int(self.v1))
		
		if (self.v2 != "" and self.v2[0] != "\""):
			if (self.v2[0] == "_"):
				y = searchVar(self.v2)
			elif (self.v2[0] == "T"):
				y = stackTemp.pop()
			else: 
				if (self.v2.find(".") != -1):
					y = SymTab("Y","float",float(self.v2))
				else: 
					y = SymTab("Y","int",int(self.v2))
		
		#---------------------#
		# Cuadruple compiling #
		#---------------------#

		# ---- Aritmetic operators ---- #

		#+ Operator handling
		if (self.op == "+"):
			if (x.varTP == "int" and y.varTP == "int"): 
				stackTemp.append(SymTab(self.d,"int",(int(y.varVal) + int(x.varVal))))
			elif (x.varTP == "float" and y.varTP == "float"): 
				stackTemp.append(SymTab(self.d,"float",(float(y.varVal) + float(x.varVal))))
			else: 
				print("Data type mismatch at cuadruple " + str(J))
		#- Operator Handling
		elif (self.op == "-"):
			if (x.varTP == "int" and y.varTP == "int"): 
				stackTemp.append(SymTab(self.d,"int",(int(y.varVal) - int(x.varVal))))
			elif (x.varTP == "float" and y.varTP == "float"): 
				stackTemp.append(SymTab(self.d,"float",(float(y.varVal) - float(x.varVal))))
			else: 
				print("Data type mismatch at cuadruple " + str(J))
		#* Operator handling
		elif (self.op == "*"):
			if (x.varTP == "int" and y.varTP == "int"): 
				stackTemp.append(SymTab(self.d,"int",(int(y.varVal) * int(x.varVal))))
			elif (x.varTP == "float" and y.varTP == "float"): 
				stackTemp.append(SymTab(self.d,"float",(float(y.varVal) * float(x.varVal))))
			else: 
				print("Data type mismatch at cuadruple " + str(J))
		#/ Operator handling
		elif (self.op == "/"):
			if (x.varTP == "int" and y.varTP == "int"): 
				stackTemp.append(SymTab(self.d,"int",(int(y.varVal) // int(x.varVal))))
			elif (x.varTP == "float" and y.varTP == "float"): 
				stackTemp.append(SymTab(self.d,"float",(float(y.varVal) / float(x.varVal))))
			else: 
				print("Data type mismatch at cuadruple " + str(J))

		# ---- Logic operators ---- #

		#> Operator handling
		elif (self.op == ">"):
			if (x.varTP == "int" and y.varTP == "int"): 
				if (int(y.varVal)>int(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			elif (x.varTP == "float" and y.varTP == "float"): 
				if (float(y.varVal)>float(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			else: 
				print("Data type mismatch at cuadruple " + str(J))
		#>= Operator handling
		elif (self.op == ">="):
			if (x.varTP == "int" and y.varTP == "int"): 
				if (int(y.varVal)>=int(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			elif (x.varTP == "float" and y.varTP == "float"): 
				if (float(y.varVal)>=float(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			else: 
				print("Data type mismatch at cuadruple " + str(J))
		#< Operator handling
		elif (self.op == "<"):
			if (x.varTP == "int" and y.varTP == "int"): 
				if (int(y.varVal)<int(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			elif (x.varTP == "float" and y.varTP == "float"): 
				if (float(y.varVal)<float(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			else: 
				print("Data type mismatch at cuadruple " + str(J))
		#<= Operator handling
		elif (self.op == "<="):
			if (x.varTP == "int" and y.varTP == "int"): 
				if (int(y.varVal)<=int(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			elif (x.varTP == "float" and y.varTP == "float"): 
				if (float(y.varVal)<=float(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			else: 
				print("Data type mismatch at cuadruple " + str(J))
		#== Operator handling
		elif (self.op == "=="):
			if (x.varTP == "int" and y.varTP == "int"): 
				if (int(y.varVal)==int(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			elif (x.varTP == "float" and y.varTP == "float"): 
				if (float(y.varVal)==float(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			else: 
				print("Data type mismatch at cuadruple " + str(J))
		#!= Operator handling
		elif (self.op == "!="):
			if (x.varTP == "int" and y.varTP == "int"): 
				if (int(y.varVal)!=int(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			elif (x.varTP == "float" and y.varTP == "float"): 
				if (float(y.varVal)!=float(x.varVal)): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			else: 
				print("Data type mismatch at cuadruple " + str(J))
		#&& And handling
		elif (self.op == "AND"):
			if (x.varTP == "bool" and y.varTP == "bool"): 
				if (y.varVal and x.varVal): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			else: 
				print("Data type mismatch at cuadruple " + str(J))
		#|| Or handling
		elif (self.op == "OR"):
			if (x.varTP == "bool" and y.varTP == "bool"): 
				if (y.varVal or x.varVal): stackTemp.append(SymTab(self.d,"bool",True))
				else: stackTemp.append(SymTab(self.d,"bool",False))
			else: 
				print("Data type mismatch at cuadruple " + str(J))

		# ---- Jumps ---- #

		#Goto False directive handling
		elif (self.op == "GF"):
			if (not x.varVal): J = int(y.varVal) - 1
		#Goto directive handling 
		elif (self.op == "G"):
			J = int(x.varVal) - 1
		#Call directive handling 
		elif (self.op == "C"):
			stackJmp.append(J)
			J = int(searchMod(self.v1 + ":").varVal) - 1 
		#Call directive handling 
		elif (self.op == "RETURN"):
			if (len(stackJmp) > 0):
				J = stackJmp.pop()

		# ---- Misc ---- #

		#Save directive handling
		elif (self.op == "S"):
			searchVar(self.v2).varVal = x.varVal
			searchVar(self.v2).varTP = x.varTP
		#Read directive handling
		elif (self.op == "R"):
			searchVar(self.v1).varVal = input()
		#Print directive handling
		elif (self.op == "P"):
			if(self.v1 == "\"n_l\""):
				print("")
			else: 
				if (self.v1[0] == "\""): 
					print (self.v1.replace("\"",""), end = "")
				else: 
					print (searchVar(self.v1).varVal, end = "")	
		
		J = J+1
		return J
		


#Funcion para hacer cuadruplos con operaciones binarias.
def cuadBin(op, v1, v2, t):
	global pc
	cd = Cuad(op, v1, v2, t)
	stackPrg.append(cd)
	pc = pc + 1

#Funcion para buscar variable por ID.
def searchVar(idv):
	lb = idv.find("[")
	rb = idv.rfind("]")
	if (lb != -1): #Si la variable es dimensionada
		varID = idv.replace(idv[lb:rb+1],'')
		dim = idv[lb+1:rb].split("][") #Extrae las dimensiones de la variable
		for d in dim:
			if (d[0] == "_"):
				d = searchVar(d).varVal
			varID = varID + "[" + str(d) + "]"
		for J in range(len(varMap)):
			if(varMap[J].varID == varID):
				return varMap[J]
	else: 	
		for J in range(len(varMap)):
			if(varMap[J].varID == idv):
				return varMap[J]

#Funcion para buscar temporales.
def searchTemp(id):
	for J in range(len(stackTemp)):
		if(stackTemp[J].varID == id):
			return stackTemp[J]

#Funcion para buscar funciones.
def searchMod(id):
	for J in range(len(stackMod)):
		if(stackMod[J].varID == id):
			return stackMod[J]

#Funcion para elegir un archivo de pruebas. 
def opF(loc):
	F = []
	numF = ''
	ans = False
	cont = 1
	
	for base, dirs, files in os.walk(loc):
		F.append(files)
	
	for file in files:
		print (str(cont) + ". "+file)
		cont = cont+1

	while ans == False: 
		numF = int(input('Input test file number: '))-1
		for file in files:
			if numF < len(files) and file == files[numF]:
				ans = True
				break 
		if ans == False: 
			print ("File not found.")
	
	print (files[numF] + " chosen.")
	return files[numF]

#Funcion para agregar variables a la tabla de simbolos
def varBuild(arrVar, vType, val):
	dim = []
	con = []
	eq = False
	for var in arrVar:
		lb = var.find("[")
		rb = var.rfind("]")
		if (lb != -1): #Si la variable es dimensionada
			varID = var.replace(var[lb:rb+1],'')
			dim = var[lb+1:rb].split("][") #Extrae las dimensiones de la variable
			for n in range(len(dim)):
				dim[n]= int(dim[n]) #Convierte las dimensiones en enteros y crea un arreglo de las mismas dimensiones con 0
				con.append(0)
			while (not eq):
				index = 0
				vdim = varID
				for i in con:
					vdim = vdim + "[" + str(i) + "]"
				x =SymTab(vdim,vType,val)
				varMap.append(x)	#Agrega la variable dimensionada a la tabla de simbolos

				con[index]=con[index]+1

				while (con[index] == dim[index]):
					con[index] = 0
					index = index + 1
					if (index == len(con)):
						eq = True
						break
					con[index]=con[index]+1
					
		else: 
			x =SymTab(var,vType,val)
			varMap.append(x)	#Agrega la varibale simple a la tabla de simbolos

######################
#Definicion de tokens#
######################
tokens = [
	#Numeros y letras
	'NUM', 'STR',

	#Tipos de variables
	'FLOAT', 'INT',

	#Operadores
	'PLUS',	'MINUS', 'TIMES', 'DIVIDE',	'LPAREN', 'RPAREN',	'LBRACK', 'RBRACK',

	#Signos de puntuacion
	'COMMA', 'SKP', 'PYC', 'DOT',

	#Simbolo de asignacion de variable
	'VASS',

	#Expresiones logicas y comparadores
	'GT', 'GTE', 'LT', 'LTE', 'EQ', 'NEQ', 'AND', 'OR',

	#IDs
	'MODULE', 'VAR', 'CON', 'ELSE', 'READ', 'PRINT', 'LOOP', 'FOR', 'ID', 'USE',
	
	#Finales
	'M_END', 'V_END', 'C_END', 'L_END', 'F_END'	
]

#Expresiones regulares para tokens simples
t_COMMA		= r'\,'
t_DOT		= r'\.'
t_PLUS		= r'\+'
t_MINUS		= r'\-'
t_TIMES		= r'\*'
t_DIVIDE	= r'\/'
t_LPAREN	= r'\('
t_RPAREN	= r'\)'
t_LBRACK	= r'\['
t_RBRACK	= r'\]'
t_PYC		= r'\;'
t_GT		= r'\>'
t_GTE		= r'\>\='
t_LT		= r'\<'
t_LTE		= r'\<\='
t_EQ		= r'\=\='
t_NEQ		= r'\!\='
t_AND		= r'\&\&'
t_OR		= r'\|\|'
t_VASS 		= r'\:\='

t_ignore 	= r' '
t_ignore_TAB= r'\t'
t_ignore_BCK= r'\r'
t_STR		= r'"([^"]*)"'

#Numeros
def t_NUM(t):
    r'[-]??\d+'
    t.value = int(t.value)    
    return t

#ID
def t_ID(t):
	r'_[a-zA-Z][a-zA-Z0-9_]*' #Acepta ids que empiecen con una letra mayuscula o minuscula seguido de letras, numeros o guiones bajos
	t.value = str(t.value)
	return t

#Permite los comentarios omitiendo todo lo que se encuentre entre colones (~)
def t_comment(t):
	r'~([^~]*)~'
	print ("Comment")
	pass

def t_SKP(t):
    r'\n'
    t.lexer.lineno += 1
    return t

#Tokens para tipos de variables 
def t_INT(t):
	r'INT'
	t.type = 'INT'
	print("Int type")
	return t

def t_FLOAT(t):
	r'FLOAT'
	t.type = 'FLOAT'
	print("Float type")
	return t

#Tokens de procedimientos
def t_USE(t):
	r'USE'
	t.type = 'USE'
	print("USE")
	return t

def t_READ(t):
	r'READ'
	t.type = 'READ'
	print("Read")
	return t

def t_PRINT(t):
	r'PRINT'
	t.type = 'PRINT'
	print("Print")
	return t

def t_MODULE(t):
	r'MODULE'
	t.type = 'MODULE'
	print("Start of module")
	return t

def t_VAR(t):
	r'VAR'
	t.type = 'VAR'
	print("Variable declaration")
	return t

def t_CON(t):
	r'CON'
	t.type = 'CON'
	print("Start of condition")
	return t

def t_ELSE(t):
	r'ELSE'
	t.type = 'ELSE'
	print("Else")
	return t

def t_LOOP(t):
	r'LOOP'
	t.type = 'LOOP'
	print("Start of loop")
	return t

def t_FOR(t):
	r'FOR'
	t.type = 'FOR'
	print("Start of for loop")
	return t

def t_M_END(t):
	r'M_END'
	t.type = 'M_END'
	print('End of module')
	return t

def t_V_END(t):
	r'V_END'
	t.type = 'V_END'
	print('End of variable declaration')
	return t

def t_C_END(t):
	r'C_END'
	t.type = 'C_END'
	print('End of conditional')
	return t

def t_L_END(t):
	r'L_END'
	t.type = 'L_END'
	print('End of loop')
	return t

def t_F_END(t):
	r'F_END'
	t.type = 'F_END'
	print('End of for loop')
	return t

#Manejo de errores
def t_error(t):
	print("Illegal character: '" + str(t.value[0]) + "'") #Muestra los caracteres ilegales
	t.lexer.skip(1)

#______________________________________#
#--------- Reglas de sintaxis ---------#
#______________________________________#
def p_S(p):
	'''
	S : VA SKP M
	'''
	print("\tSINTAX: OK\n\n\tSYMBOL TABLE: \n")		#Se ejecuta cuando la sintaxis es correcta e imprime la estructura que almacena la informacion de las variables
	for J in range(len(varMap)):
		print("LOC: " + str(J), end ="\t")
		varMap[J].out()
	print("\n\tCUADRUPLOS: \n")
	for J in range(len(stackPrg)):
		print(J, end = "\t")
		stackPrg[J].out()
	print("\n\tPROGRAM EXECUTION: \n")
	J = 0
	for m in stackMod:
		if m.varID == "_MAIN:": 
			J = m.varVal
			break
	while (J < len(stackPrg)):
		J = stackPrg[J].exe(J)

# DECLARACION DE VARIABLES #
def p_VA(p):
	'''
	VA 	: VAR SKP VX V_END
	VX	: VX VY
		| VY
	VY	: VFD
		| VFU
		| VID
		| VIU
		| EMPTY
	'''

def p_VFD(p):
	'VFD : FLOAT VAX VASS NUM DOT NUM SKP'
	varS = str(p[2])
	varV = str(p[4]) + "." + str(p[6])
	arrVar = varS.split(",")
	varBuild(arrVar, "float", varV)

def p_VFU(p):
	'VFU : FLOAT VAX SKP'
	varS = str(p[2])
	arrVar = varS.split(",")
	varBuild(arrVar, "float", "0.0")

def p_VID(p):
	'VID : INT VAX VASS NUM SKP'
	varS = str(p[2])
	varV = str(p[4])
	arrVar = varS.split(",")
	varBuild(arrVar, "int", varV)

def p_VIU(p):
	'VIU : INT VAX SKP'
	varS = str(p[2])
	arrVar = varS.split(",")
	varBuild(arrVar, "int", "0")		

def p_VAX_COMMA(p):
	'VAX : VAX COMMA VAA'
	p[0] = p[1] + "," + p[3]

def p_VAX(p):
	'VAX : VAA'
	p[0] = p[1]

def p_VAA_ID(p):
	'VAA : ID'
	p[0] = p[1] 

def p_VAA_ID_DIM(p):
	'VAA : ID DIM'
	varS = str(p[2])
	sizes = varS.split(",")
	p[0] = p[1] + p[2]

def p_DIM(p):
	'DIM : DIM DIMY'
	p[0] = str(p[1]) + str(p[2])

def p_DIM_DY(p):
	'DIM : DIMY'
	p[0]=p[1]

def p_DIMY(p):
	'DIMY : LBRACK NID RBRACK'
	p[0] = p[1] + str(p[2]) + p[3]

# DEFINICION DE MODULOS #
def p_M(p):
	'''
	M	: MX SKP M
		| EMPTY
	'''

def p_MX(p):
	'MX	: MODULE MID SKP A MY'

def p_MID(p): 
	'MID : ID'
	stackMod.append(SymTab(p[1] + ":","mod",pc))
	cuadBin(p[1]+":",str(pc),"","")

def p_MY(p):
	'MY : M_END'
	cuadBin("RETURN","","","")

# PROCEDIMIENTOS #
def p_A(p):
	'''
	A	: A AX SKP
		| EMPTY
	'''

def p_AX(p):
	'''
	AX	: RD
	 	| PR
		| CD
		| LP
		| FR
		| AR
		| US
		| EMPTY
	'''

# IMPRESION #
def p_PR(p):
	'PR	: PRINT LPAREN PX RPAREN'

def p_PX(p):
	'PX	: PX PLUS PX'

def p_PX_STR(p):
	'PX	: STR'
	cuadBin("P",p[1],"","") 

def p_PX_ID(p):
	'PX	: VAA'
	cuadBin("P",p[1],"","")	

# LECTURA #
def p_RD(p):
	'RD	: READ LPAREN VAA RPAREN'
	x = searchVar(p[3])
	cuadBin("R",p[3],"","")

# LLAMADO A MODULOS #
def p_US(p):
	'US	: USE ID'
	cuadBin("C", p[2],"","")

# CONDICIONAL #
def p_CD(p):
	'''
	CD	: CON EL CDX SKP A CDE CDZ
	CDE : CDY SKP A
		| EMPTY
	'''

def p_CDX(p):
	'CDX : PYC'
	r = stackOp.pop()
	cuadBin("GF",r,"","")
	stackJmp.append(pc-1)
	
def p_CDY(p):
	'CDY : ELSE'
	cuadBin("G","","","") 
	stackPrg[stackJmp.pop()].v2 = str(pc)
	stackJmp.append(pc-1)

def p_CDZ(p):
	'CDZ : C_END'
	sjp = stackJmp.pop()
	if (stackPrg[sjp].op == "GF"):
		stackPrg[sjp].v2 = str(pc)
	else:
		stackPrg[sjp].v1 = str(pc)
	

# WHILE #
def p_LP(p):
	'LP	: LOOP EL LPX SKP A LPY'

def p_LPX(p):
	'LPX : PYC'
	r = stackOp.pop()
	cuadBin("GF",r,"","")
	stackJmp.append(pc-1)

def p_LPY(p):
	'LPY : L_END' 
	r = stackJmp.pop()
	cuadBin("G",str(r-1),"","")
	stackPrg[r].v2 = str(pc)

# FOR #
def p_FR(p):
	'FR	: FOR FRX SKP A FRY'

def p_FRX(p):
	'FRX : ID VASS NID SL NID COMMA NID PYC'
	global tc
	tc = tc + 1
	t = "T"+str(tc)
	x = searchVar(p[1])
	stackOp.append(p[1])
	stackOp.append(p[7])
	cuadBin("S",str(p[3]),str(p[1]),"")
	cuadBin(str(p[4]),str(p[5]),str(p[1]),t)
	cuadBin("GF",t,"","")
	stackJmp.append(pc-1)

def p_FRY(p):
	'FRY : F_END'
	global tc
	tc = tc + 1
	t = "T" + str(tc)
	r = stackJmp.pop()
	a = stackOp.pop()
	b = stackOp.pop()
	cuadBin("+",str(a),str(b),t)
	cuadBin("S",t,str(b),"")
	cuadBin("G",str(r-1),"","")
	stackPrg[r].v2 = str(pc)

# EXPRESIONES ARITMETICAS #
def p_AR(p):
	'AR	: VAA VASS EA'
	cuadBin("S",stackOp.pop(),p[1],"")

def p_EA_PLUS(p):
	'EA	: EA PLUS TA'
	global tc
	tc = tc + 1
	t = "T"+str(tc)
	cuadBin("+",stackOp.pop(),stackOp.pop(),t)
	stackOp.append(t)

def p_EA_MINUS(p):
	'EA	: EA MINUS TA'
	global tc
	tc = tc + 1
	t = "T"+str(tc)
	cuadBin("-",stackOp.pop(),stackOp.pop(),t)
	stackOp.append(t)
	
def p_EA_TA(p):
	'EA	:	TA'

def p_TA_TIMES(p):
	'TA	: TA TIMES FA'
	global tc 
	tc = tc + 1
	t = "T"+str(tc)
	cuadBin("*",stackOp.pop(),stackOp.pop(),t)
	stackOp.append(t)

def p_TA_DIVIDE(p):
	'TA	: TA DIVIDE FA'
	global tc 
	tc = tc + 1
	t = "T"+str(tc)
	cuadBin("/",stackOp.pop(),stackOp.pop(),t)
	stackOp.append(t)

def p_TA_FA(p):
	'TA	: FA'

def p_FA(p):
	'FA	: NID'
	stackOp.append(str(p[1]))

def p_FA_PAREN(p):
	'FA	: LPAREN EA RPAREN'

# EXPRESIONES LOGICAS #
def p_EL_OR(p):
	'EL	: EL OR TL'
	global tc
	tc = tc + 1
	t = "T"+str(tc)
	cuadBin("OR",stackOp.pop(),stackOp.pop(),t)
	stackOp.append(t)

def p_EL(p):
	'EL	: TL'

def p_TL_AND(p):
	'TL	: TL AND FL'
	global tc
	tc = tc + 1
	t = "T"+str(tc)
	cuadBin("AND",stackOp.pop(),stackOp.pop(),t)
	stackOp.append(t)

def p_TL(p):
	'TL	: FL'

def p_FL_SL(p):
	'FL	: FL SL PL'
	global tc
	tc = tc + 1
	t = "T"+str(tc)
	cuadBin(str(p[2]),stackOp.pop(),stackOp.pop(),t)
	stackOp.append(t)

def p_FL(p):
	'FL	: PL'

def p_PL(p):
	'PL	: NID'
	stackOp.append(str(p[1]))

def p_PL_PAREN(p):
	'PL : LPAREN EL RPAREN'

#Simbolos logicos y operandos
def p_SL(p):
	'''
	SL	: GT 
		| GTE
		| LT 
		| LTE 
		| EQ 
		| NEQ
	'''
	p[0]=p[1]

def p_NID(p):
	'''
	NID	: VAA
		| NUM
		| NUM DOT NUM
	'''
	if len(p) > 2:
		p[0] = str(p[1]) + "." + str(p[3])
	else: 	
		p[0] = str(p[1])

#Expresion vacia
def p_EMPTY(p):
	'EMPTY :'
	pass

#Mensaje de error
def p_error(p):
	print("\tSyntax error")

###############################
# PRUEBA DE LEXICO Y SINTAXIS #
###############################
def main():
	
	arch = opF(loc)
	t = loc + arch 
	fp = codecs.open(t,"r","utf-8")
	data = fp.read()
	fp.close()
	
	lexer = lex.lex(debug=0)
	lexer.input(data)
	while True:
		tok = lexer.token()
		if not tok:
			break      # No more input
		print(tok)
	parser = yacc.yacc()
	parser.parse(data)

if __name__ == '__main__':
    main()