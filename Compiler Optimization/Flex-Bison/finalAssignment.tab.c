/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     T_COMMA = 258,
     T_SEMICOL = 259,
     T_OPENKEY = 260,
     T_CLOSEKEY = 261,
     T_OPENBRACK = 262,
     T_CLOSEBRACK = 263,
     T_OPENPAR = 264,
     T_CLOSEPAR = 265,
     T_PLUSOP = 266,
     T_MULTOP = 267,
     T_MINUSOP = 268,
     T_DIVOP = 269,
     T_ASSOP = 270,
     T_EQSYM = 271,
     T_LESSOP = 272,
     T_MOREOP = 273,
     T_LESSOEQOP = 274,
     T_MOREOEQOP = 275,
     T_MODOP = 276,
     T_POWOP = 277,
     T_COLON = 278,
     T_READ = 279,
     T_PRINT = 280,
     T_DOIF = 281,
     T_DOIFELSE = 282,
     T_DOWHILE = 283,
     T_DOUNTIL = 284,
     T_PROGRAM = 285,
     T_BEGIN = 286,
     T_END = 287,
     T_INT = 288,
     T_FLT = 289,
     T_TENSOR = 290,
     T_MATMUL = 291,
     T_TRANSPOSE = 292,
     T_REDUCE = 293,
     T_RESHAPE = 294,
     T_ID = 295,
     T_INTEGER = 296,
     T_FLOAT = 297
   };
#endif
/* Tokens.  */
#define T_COMMA 258
#define T_SEMICOL 259
#define T_OPENKEY 260
#define T_CLOSEKEY 261
#define T_OPENBRACK 262
#define T_CLOSEBRACK 263
#define T_OPENPAR 264
#define T_CLOSEPAR 265
#define T_PLUSOP 266
#define T_MULTOP 267
#define T_MINUSOP 268
#define T_DIVOP 269
#define T_ASSOP 270
#define T_EQSYM 271
#define T_LESSOP 272
#define T_MOREOP 273
#define T_LESSOEQOP 274
#define T_MOREOEQOP 275
#define T_MODOP 276
#define T_POWOP 277
#define T_COLON 278
#define T_READ 279
#define T_PRINT 280
#define T_DOIF 281
#define T_DOIFELSE 282
#define T_DOWHILE 283
#define T_DOUNTIL 284
#define T_PROGRAM 285
#define T_BEGIN 286
#define T_END 287
#define T_INT 288
#define T_FLT 289
#define T_TENSOR 290
#define T_MATMUL 291
#define T_TRANSPOSE 292
#define T_REDUCE 293
#define T_RESHAPE 294
#define T_ID 295
#define T_INTEGER 296
#define T_FLOAT 297




/* Copy the first part of user declarations.  */
#line 3 "finalAssignment.y"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "types.h"
#include "tensor_graph.h"
#include "optimizer.h"
#include "cuda_gen.h"
#include "geometry.h"
#include "llvm_lowering.h"

int  yylex(void);
int  yyerror(const char *s);

extern char *yytext;
extern FILE *yyin;
int        line = 1, order = 1;

/* Global AST root & symbol table */
ARST      *rootAST     = NULL;
HashTable *symbolTable = NULL;
TensorGraph *tensorGraph = NULL;  /* Tensor operation graph */

/* --- AST node --- */
struct arst {
    unsigned char type;   /* 0=internal,1=leaf */
    int           token;
    char         *valStr; /* for T_ID */
    int           valInt; /* for literals & stored ints */
    float         valFlt; /* for literals & stored floats */
    ARST         *left, *right;
};

/* --- symbol‐table structs --- */
struct Ht_item {
    char  *key;
    ARST  *treeNode;
};
struct LinkedList {
    Ht_item          *item;
    struct LinkedList *next;
};
struct HashTable {
    Ht_item      **items;
    LinkedList   **overflow_buckets;
    int            size, count;
};

/* Forward declarations of helpers */
ARST      *newNode(unsigned char,int,char*,int,float,ARST*,ARST*);
void       execAST(ARST*);
float      evalF(ARST*);
int        evalCond(ARST*);
float      potenciaConWhile(float,int);
HashTable *create_table(void);
void       ht_insert(HashTable*,char*,ARST*);
ARST      *ht_search(HashTable*,char*);
void       printTreeLevels(ARST*);
void       print_table(HashTable*);
void       free_table(HashTable*);


/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 67 "finalAssignment.y"
{
  ARST  *ast;
  char  *id;
  int    ival;
  float  fval;
}
/* Line 193 of yacc.c.  */
#line 250 "finalAssignment.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 263 "finalAssignment.tab.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  5
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   116

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  43
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  18
/* YYNRULES -- Number of rules.  */
#define YYNRULES  52
/* YYNRULES -- Number of states.  */
#define YYNSTATES  122

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   297

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint8 yyprhs[] =
{
       0,     0,     3,     5,    12,    14,    15,    19,    22,    24,
      28,    30,    34,    36,    38,    40,    42,    48,    52,    54,
      58,    62,    70,    81,    89,    97,   100,   103,   105,   106,
     110,   113,   117,   121,   123,   127,   131,   135,   139,   141,
     145,   147,   149,   151,   155,   159,   163,   167,   171,   175,
     179,   183,   188
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      44,     0,    -1,    45,    -1,    30,    40,    46,    31,    54,
      32,    -1,    47,    -1,    -1,    48,     4,    47,    -1,    48,
       4,    -1,    48,    -1,    50,    23,    49,    -1,    51,    -1,
      40,     3,    49,    -1,    40,    -1,    33,    -1,    34,    -1,
      35,    -1,    35,    40,     7,    52,     8,    -1,    41,     3,
      52,    -1,    41,    -1,    40,    15,    56,    -1,    40,    15,
      60,    -1,    26,     9,    59,    10,     7,    54,     8,    -1,
      27,     9,    59,    10,     7,    54,     8,     7,    54,     8,
      -1,    28,     9,    59,    10,     7,    54,     8,    -1,    29,
       9,    59,    10,     7,    54,     8,    -1,    25,    56,    -1,
      24,    40,    -1,    55,    -1,    -1,    53,     4,    55,    -1,
      53,     4,    -1,    56,    11,    57,    -1,    56,    13,    57,
      -1,    57,    -1,    57,    12,    58,    -1,    57,    14,    58,
      -1,    57,    21,    58,    -1,    57,    22,    58,    -1,    58,
      -1,     9,    56,    10,    -1,    40,    -1,    42,    -1,    41,
      -1,    56,    17,    56,    -1,    56,    18,    56,    -1,    56,
      19,    56,    -1,    56,    20,    56,    -1,    56,    16,    56,
      -1,    40,    36,    40,    -1,    40,    11,    40,    -1,    40,
      12,    40,    -1,    37,     9,    40,    10,    -1,    38,     9,
      40,     3,    41,    10,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   102,   102,   140,   148,   149,   153,   154,   155,   159,
     162,   168,   174,   182,   183,   184,   188,   215,   228,   237,
     243,   251,   255,   261,   265,   269,   272,   279,   280,   284,
     285,   290,   291,   292,   296,   297,   298,   299,   300,   304,
     305,   310,   311,   315,   316,   317,   318,   319,   323,   342,
     356,   370,   384
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "T_COMMA", "T_SEMICOL", "T_OPENKEY",
  "T_CLOSEKEY", "T_OPENBRACK", "T_CLOSEBRACK", "T_OPENPAR", "T_CLOSEPAR",
  "T_PLUSOP", "T_MULTOP", "T_MINUSOP", "T_DIVOP", "T_ASSOP", "T_EQSYM",
  "T_LESSOP", "T_MOREOP", "T_LESSOEQOP", "T_MOREOEQOP", "T_MODOP",
  "T_POWOP", "T_COLON", "T_READ", "T_PRINT", "T_DOIF", "T_DOIFELSE",
  "T_DOWHILE", "T_DOUNTIL", "T_PROGRAM", "T_BEGIN", "T_END", "T_INT",
  "T_FLT", "T_TENSOR", "T_MATMUL", "T_TRANSPOSE", "T_REDUCE", "T_RESHAPE",
  "T_ID", "T_INTEGER", "T_FLOAT", "$accept", "start_program", "prog",
  "opt_decls", "decl_lst", "decl", "id_list", "type", "tensor_decl",
  "tensor_dims", "stmt", "opt_stmts", "stmt_lst", "expr", "term", "factor",
  "expression", "tensor_op", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    43,    44,    45,    46,    46,    47,    47,    47,    48,
      48,    49,    49,    50,    50,    50,    51,    52,    52,    53,
      53,    53,    53,    53,    53,    53,    53,    54,    54,    55,
      55,    56,    56,    56,    57,    57,    57,    57,    57,    58,
      58,    58,    58,    59,    59,    59,    59,    59,    60,    60,
      60,    60,    60
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     6,     1,     0,     3,     2,     1,     3,
       1,     3,     1,     1,     1,     1,     5,     3,     1,     3,
       3,     7,    10,     7,     7,     2,     2,     1,     0,     3,
       2,     3,     3,     1,     3,     3,     3,     3,     1,     3,
       1,     1,     1,     3,     3,     3,     3,     3,     3,     3,
       3,     4,     6
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     2,     5,     1,    13,    14,    15,     0,
       4,     8,     0,    10,     0,    28,     7,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    27,     6,
      12,     9,    18,     0,    26,     0,    40,    42,    41,    25,
      33,    38,     0,     0,     0,     0,     0,    30,     3,     0,
       0,    16,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    40,    19,    20,    29,
      11,    17,    39,    31,    32,    34,    35,    36,    37,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    47,    43,    44,    45,    46,    28,    28,
      28,    28,     0,     0,    49,    50,    48,     0,     0,     0,
       0,    51,     0,    21,     0,    23,    24,     0,    28,    52,
       0,    22
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,     2,     3,     9,    10,    11,    31,    12,    13,    33,
      26,    27,    28,    59,    40,    41,    60,    68
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -94
static const yytype_int8 yypact[] =
{
     -13,   -16,    28,   -94,    22,   -94,   -94,   -94,     5,     0,
     -94,    54,    44,   -94,    61,    14,    22,    29,    30,    32,
      -5,    64,    65,    66,    67,    55,    73,    46,   -94,   -94,
      76,   -94,    77,    74,   -94,    -5,   -94,   -94,   -94,    -1,
      -3,   -94,    -5,    -5,    -5,    -5,    -8,    14,   -94,    29,
      30,   -94,     3,    -5,    -5,    -5,    -5,    -5,    -5,    33,
      71,    75,    78,    79,    81,    82,    -9,    -1,   -94,   -94,
     -94,   -94,   -94,    -3,    -3,   -94,   -94,   -94,   -94,    -5,
      -5,    -5,    -5,    -5,    80,    85,    86,    87,    43,    56,
      57,    58,    59,    -1,    -1,    -1,    -1,    -1,    14,    14,
      14,    14,    90,    83,   -94,   -94,   -94,    93,    94,    95,
      96,   -94,    68,   -94,    88,   -94,   -94,    97,    14,   -94,
      98,   -94
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -94,   -94,   -94,   -94,    89,   -94,    35,   -94,   -94,    60,
     -94,   -93,    69,   -20,    -6,   -35,    21,   -94
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint8 yytable[] =
{
      39,    35,    90,    91,    35,   107,   108,   109,   110,    55,
      53,    56,    54,    72,    53,    52,    54,     1,    57,    58,
      75,    76,    77,    78,     4,   120,    67,    92,     5,    64,
      65,    15,    66,    37,    38,    36,    37,    38,    19,    20,
      21,    22,    23,    24,    53,    14,    54,    73,    74,    79,
      80,    81,    82,    83,    25,     6,     7,     8,    16,    93,
      94,    95,    96,    97,    61,    62,    63,    17,    18,    30,
      46,    32,    34,    42,    43,    44,    45,    47,    48,    49,
      50,    84,    51,   102,    70,    85,   112,    98,    86,    87,
      88,    89,    99,   100,   101,   118,   103,   104,   105,   106,
     111,   113,   114,   115,   116,    29,   121,   119,     0,   117,
      71,     0,     0,     0,     0,     0,    69
};

static const yytype_int8 yycheck[] =
{
      20,     9,    11,    12,     9,    98,    99,   100,   101,    12,
      11,    14,    13,    10,    11,    35,    13,    30,    21,    22,
      55,    56,    57,    58,    40,   118,    46,    36,     0,    37,
      38,    31,    40,    41,    42,    40,    41,    42,    24,    25,
      26,    27,    28,    29,    11,    40,    13,    53,    54,    16,
      17,    18,    19,    20,    40,    33,    34,    35,     4,    79,
      80,    81,    82,    83,    43,    44,    45,    23,     7,    40,
      15,    41,    40,     9,     9,     9,     9,     4,    32,     3,
       3,    10,     8,    40,    49,    10,     3,     7,    10,    10,
       9,     9,     7,     7,     7,     7,    40,    40,    40,    40,
      10,     8,     8,     8,     8,    16,     8,    10,    -1,    41,
      50,    -1,    -1,    -1,    -1,    -1,    47
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    30,    44,    45,    40,     0,    33,    34,    35,    46,
      47,    48,    50,    51,    40,    31,     4,    23,     7,    24,
      25,    26,    27,    28,    29,    40,    53,    54,    55,    47,
      40,    49,    41,    52,    40,     9,    40,    41,    42,    56,
      57,    58,     9,     9,     9,     9,    15,     4,    32,     3,
       3,     8,    56,    11,    13,    12,    14,    21,    22,    56,
      59,    59,    59,    59,    37,    38,    40,    56,    60,    55,
      49,    52,    10,    57,    57,    58,    58,    58,    58,    16,
      17,    18,    19,    20,    10,    10,    10,    10,     9,     9,
      11,    12,    36,    56,    56,    56,    56,    56,     7,     7,
       7,     7,    40,    40,    40,    40,    40,    54,    54,    54,
      54,    10,     3,     8,     8,     8,     8,    41,     7,    10,
      54,     8
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 102 "finalAssignment.y"
    {
      /* (*) symbolTable has already been created in main() before yyparse() */
      rootAST = (yyvsp[(1) - (1)].ast);
      execAST(rootAST);
      printf("\n=== Execution tree by levels ===\n");
      printTreeLevels(rootAST);
      print_table(symbolTable);
      
      /* Process tensor operations if graph exists */
      if (tensorGraph && tensorGraph->num_nodes > 0) {
        print_graph(tensorGraph);
        
        /* Apply optimizations */
        OptStrategy strategies[] = {OPT_FUSE_OPS, OPT_COMMON_SUBEXPR, OPT_MEMORY_LAYOUT};
        OptResult *opt_result = optimize_graph(tensorGraph, strategies, 3);
        print_optimization_report(opt_result);
        
        /* Generate CUDA code */
        generate_cuda_code(tensorGraph, "generated_kernels.cu");
        printf("\nCUDA code generated in: generated_kernels.cu\n");
        
        /* Generate LLVM IR for analysis */
        if (lower_graph_to_llvm_ir(tensorGraph, "tensor_output.ll") == 0) {
            printf("LLVM IR generated in: tensor_output.ll\n");
            printf("Run './llvm_analyze.sh tensor_output.ll' for detailed analysis\n");
        } else {
            fprintf(stderr, "Warning: LLVM IR generation failed\n");
        }
        
        free_opt_result(opt_result);
      }
      
      if (tensorGraph) free_tensor_graph(tensorGraph);
      free_table(symbolTable);
    ;}
    break;

  case 3:
#line 140 "finalAssignment.y"
    {
      ARST *n = newNode(0, T_PROGRAM, (yyvsp[(2) - (6)].id), 0, 0.0f, (yyvsp[(3) - (6)].ast), (yyvsp[(5) - (6)].ast));
      ht_insert(symbolTable, (yyvsp[(2) - (6)].id), n);
      (yyval.ast) = n;
    ;}
    break;

  case 4:
#line 148 "finalAssignment.y"
    { (yyval.ast) = (yyvsp[(1) - (1)].ast); ;}
    break;

  case 5:
#line 149 "finalAssignment.y"
    { (yyval.ast) = NULL; ;}
    break;

  case 6:
#line 153 "finalAssignment.y"
    { (yyval.ast) = newNode(0,0,"decl_list",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 7:
#line 154 "finalAssignment.y"
    { (yyval.ast) = newNode(0,0,"decl",     0,0.0f,NULL,(yyvsp[(1) - (2)].ast)); ;}
    break;

  case 8:
#line 155 "finalAssignment.y"
    { (yyval.ast) = newNode(0,0,"decl",     0,0.0f,NULL,(yyvsp[(1) - (1)].ast)); ;}
    break;

  case 9:
#line 159 "finalAssignment.y"
    {
      (yyval.ast) = newNode(0,0,"decl",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast));
    ;}
    break;

  case 10:
#line 162 "finalAssignment.y"
    {
      (yyval.ast) = (yyvsp[(1) - (1)].ast);
    ;}
    break;

  case 11:
#line 168 "finalAssignment.y"
    {
      /* always insert the *leaf* ID node, not the internal list node */
      ARST *leaf = newNode(1, T_ID, (yyvsp[(1) - (3)].id), 0, 0.0f, NULL, NULL);
      ht_insert(symbolTable, (yyvsp[(1) - (3)].id), leaf);
      (yyval.ast) = leaf; /* propagate the leaf */
    ;}
    break;

  case 12:
#line 174 "finalAssignment.y"
    {
      ARST *leaf = newNode(1, T_ID, (yyvsp[(1) - (1)].id), 0, 0.0f, NULL, NULL);
      ht_insert(symbolTable, (yyvsp[(1) - (1)].id), leaf);
      (yyval.ast) = leaf;
    ;}
    break;

  case 13:
#line 182 "finalAssignment.y"
    { (yyval.ast) = newNode(0, T_INT,   "type_int",   0,0.0f, NULL, NULL); ;}
    break;

  case 14:
#line 183 "finalAssignment.y"
    { (yyval.ast) = newNode(0, T_FLT,   "type_float", 0,0.0f, NULL, NULL); ;}
    break;

  case 15:
#line 184 "finalAssignment.y"
    { (yyval.ast) = newNode(0, T_TENSOR, "type_tensor", 0,0.0f, NULL, NULL); ;}
    break;

  case 16:
#line 188 "finalAssignment.y"
    {
      /* Create tensor graph if it doesn't exist */
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      /* Parse dimensions */
      int *dims = (int*)(yyvsp[(4) - (5)].ast);
      int ndims = 0;
      int *temp = dims;
      while (*temp != -1) { ndims++; temp++; }
      
      TensorDim *td = create_tensor_dim(dims, ndims);
      GraphNode *node = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[(2) - (5)].id), td);
      
      /* Convert to iteration space */
      IterationSpace *space = tensor_to_iteration_space(td);
      printf("Tensor %s: ", (yyvsp[(2) - (5)].id));
      print_iteration_space(space);
      free_iteration_space(space);
      
      /* Free the dimensions array */
      free(dims);
      
      (yyval.ast) = newNode(0, T_TENSOR, (yyvsp[(2) - (5)].id), 0, 0.0f, NULL, NULL);
    ;}
    break;

  case 17:
#line 215 "finalAssignment.y"
    {
      /* Build dimension list - max 10 dimensions */
      int *prev_dims = (int*)(yyvsp[(3) - (3)].ast);
      int *dims = malloc(11 * sizeof(int)); /* Max 10 dims + sentinel */
      int i = 0;
      while (prev_dims[i] != -1 && i < 10) {
        dims[i] = prev_dims[i];
        i++;
      }
      dims[i] = (yyvsp[(1) - (3)].ival);
      dims[i + 1] = -1;
      (yyval.ast) = (ARST*)dims;
    ;}
    break;

  case 18:
#line 228 "finalAssignment.y"
    {
      int *dims = malloc(11 * sizeof(int));
      dims[0] = (yyvsp[(1) - (1)].ival);
      dims[1] = -1;
      (yyval.ast) = (ARST*)dims;
    ;}
    break;

  case 19:
#line 237 "finalAssignment.y"
    {
    ARST *n = newNode(0, T_ASSOP, (yyvsp[(1) - (3)].id), 0, 0.0f, NULL, (yyvsp[(3) - (3)].ast));
    ht_insert(symbolTable, (yyvsp[(1) - (3)].id), n);
    (yyval.ast) = n;
  ;}
    break;

  case 20:
#line 243 "finalAssignment.y"
    {
    if (!tensorGraph) tensorGraph = create_tensor_graph();

    /* Create assignment node in graph */
    GraphNode *node = add_graph_node(tensorGraph, OP_ASSIGN, (yyvsp[(1) - (3)].id), NULL);

    (yyval.ast) = newNode(0, T_ASSOP, (yyvsp[(1) - (3)].id), 0, 0.0f, NULL, NULL);
  ;}
    break;

  case 21:
#line 252 "finalAssignment.y"
    {
      (yyval.ast) = newNode(0, T_DOIF, "do-if", 0,0.0f, (yyvsp[(3) - (7)].ast), (yyvsp[(6) - (7)].ast));
    ;}
    break;

  case 22:
#line 257 "finalAssignment.y"
    {
      ARST *eb = newNode(0,0,"else",0,0.0f,(yyvsp[(6) - (10)].ast),(yyvsp[(9) - (10)].ast));
      (yyval.ast) = newNode(0, T_DOIFELSE, "do-ifelse", 0,0.0f, (yyvsp[(3) - (10)].ast), eb);
    ;}
    break;

  case 23:
#line 262 "finalAssignment.y"
    {
      (yyval.ast) = newNode(0, T_DOWHILE, "do-while", 0,0.0f, (yyvsp[(3) - (7)].ast), (yyvsp[(6) - (7)].ast));
    ;}
    break;

  case 24:
#line 266 "finalAssignment.y"
    {
      (yyval.ast) = newNode(0, T_DOUNTIL, "do-until", 0,0.0f, (yyvsp[(3) - (7)].ast), (yyvsp[(6) - (7)].ast));
    ;}
    break;

  case 25:
#line 269 "finalAssignment.y"
    {
      (yyval.ast) = newNode(0, T_PRINT, "print", 0,0.0f, NULL, (yyvsp[(2) - (2)].ast));
    ;}
    break;

  case 26:
#line 272 "finalAssignment.y"
    {
      ARST *idn = newNode(1, T_ID, (yyvsp[(2) - (2)].id), 0,0.0f, NULL, NULL);
      (yyval.ast) = newNode(0, T_READ, "read", 0,0.0f, idn, NULL);
    ;}
    break;

  case 27:
#line 279 "finalAssignment.y"
    { (yyval.ast) = (yyvsp[(1) - (1)].ast); ;}
    break;

  case 28:
#line 280 "finalAssignment.y"
    { (yyval.ast) = NULL; ;}
    break;

  case 29:
#line 284 "finalAssignment.y"
    { (yyval.ast) = newNode(0,0,"stmt_lst",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 30:
#line 285 "finalAssignment.y"
    { (yyval.ast) = (yyvsp[(1) - (2)].ast); ;}
    break;

  case 31:
#line 290 "finalAssignment.y"
    { (yyval.ast) = newNode(0,T_PLUSOP, "add",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 32:
#line 291 "finalAssignment.y"
    { (yyval.ast) = newNode(0,T_MINUSOP,"sub",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 33:
#line 292 "finalAssignment.y"
    { (yyval.ast) = (yyvsp[(1) - (1)].ast); ;}
    break;

  case 34:
#line 296 "finalAssignment.y"
    { (yyval.ast) = newNode(0,T_MULTOP,"mul",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 35:
#line 297 "finalAssignment.y"
    { (yyval.ast) = newNode(0,T_DIVOP,"div",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 36:
#line 298 "finalAssignment.y"
    { (yyval.ast) = newNode(0,T_MODOP,"mod",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 37:
#line 299 "finalAssignment.y"
    { (yyval.ast) = newNode(0,T_POWOP,"pow",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 38:
#line 300 "finalAssignment.y"
    { (yyval.ast) = (yyvsp[(1) - (1)].ast); ;}
    break;

  case 39:
#line 304 "finalAssignment.y"
    { (yyval.ast) = (yyvsp[(2) - (3)].ast); ;}
    break;

  case 40:
#line 305 "finalAssignment.y"
    {
      ARST *n = newNode(1, T_ID, (yyvsp[(1) - (1)].id), 0, 0.0f, NULL, NULL);
      ht_insert(symbolTable, (yyvsp[(1) - (1)].id), n);
      (yyval.ast) = n;
    ;}
    break;

  case 41:
#line 310 "finalAssignment.y"
    { (yyval.ast) = newNode(1, T_FLOAT, NULL, 0, (yyvsp[(1) - (1)].fval), NULL, NULL); ;}
    break;

  case 42:
#line 311 "finalAssignment.y"
    { (yyval.ast) = newNode(1, T_INTEGER, NULL, (yyvsp[(1) - (1)].ival), 0.0f, NULL, NULL); ;}
    break;

  case 43:
#line 315 "finalAssignment.y"
    { (yyval.ast) = newNode(0,T_LESSOP,   "lt",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 44:
#line 316 "finalAssignment.y"
    { (yyval.ast) = newNode(0,T_MOREOP,   "gt",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 45:
#line 317 "finalAssignment.y"
    { (yyval.ast) = newNode(0,T_LESSOEQOP,"le",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 46:
#line 318 "finalAssignment.y"
    { (yyval.ast) = newNode(0,T_MOREOEQOP,"ge",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 47:
#line 319 "finalAssignment.y"
    { (yyval.ast) = newNode(0,T_EQSYM,    "eq",0,0.0f,(yyvsp[(1) - (3)].ast),(yyvsp[(3) - (3)].ast)); ;}
    break;

  case 48:
#line 323 "finalAssignment.y"
    {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      /* Create matmul operation in graph */
      int dims1[] = {100, 50}; /* Example - would get from symbol table */
      int dims2[] = {50, 200};
      TensorDim *td1 = create_tensor_dim(dims1, 2);
      TensorDim *td2 = create_tensor_dim(dims2, 2);
      int out_dims[] = {100, 200};
      TensorDim *td_out = create_tensor_dim(out_dims, 2);
      
      GraphNode *in1 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[(1) - (3)].id), td1);
      GraphNode *in2 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[(3) - (3)].id), td2);
      GraphNode *op = add_graph_node(tensorGraph, OP_MATMUL, NULL, td_out);
      add_edge(in1, op);
      add_edge(in2, op);
      
      (yyval.ast) = newNode(0, T_MATMUL, "matmul", 0, 0.0f, NULL, NULL);
    ;}
    break;

  case 49:
#line 342 "finalAssignment.y"
    {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      int dims[] = {100, 100};
      TensorDim *td = create_tensor_dim(dims, 2);
      
      GraphNode *in1 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[(1) - (3)].id), td);
      GraphNode *in2 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[(3) - (3)].id), td);
      GraphNode *op = add_graph_node(tensorGraph, OP_ADD, NULL, td);
      add_edge(in1, op);
      add_edge(in2, op);
      
      (yyval.ast) = newNode(0, T_PLUSOP, "tensor_add", 0, 0.0f, NULL, NULL);
    ;}
    break;

  case 50:
#line 356 "finalAssignment.y"
    {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      int dims[] = {100, 100};
      TensorDim *td = create_tensor_dim(dims, 2);
      
      GraphNode *in1 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[(1) - (3)].id), td);
      GraphNode *in2 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[(3) - (3)].id), td);
      GraphNode *op = add_graph_node(tensorGraph, OP_MUL, NULL, td);
      add_edge(in1, op);
      add_edge(in2, op);
      
      (yyval.ast) = newNode(0, T_MULTOP, "tensor_mul", 0, 0.0f, NULL, NULL);
    ;}
    break;

  case 51:
#line 370 "finalAssignment.y"
    {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      int dims[] = {100, 50};
      TensorDim *td_in = create_tensor_dim(dims, 2);
      int out_dims[] = {50, 100};
      TensorDim *td_out = create_tensor_dim(out_dims, 2);
      
      GraphNode *in = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[(3) - (4)].id), td_in);
      GraphNode *op = add_graph_node(tensorGraph, OP_TRANSPOSE, NULL, td_out);
      add_edge(in, op);
      
      (yyval.ast) = newNode(0, T_TRANSPOSE, "transpose", 0, 0.0f, NULL, NULL);
    ;}
    break;

  case 52:
#line 384 "finalAssignment.y"
    {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      int dims[] = {100, 100};
      TensorDim *td_in = create_tensor_dim(dims, 2);
      int out_dims[] = {100};
      TensorDim *td_out = create_tensor_dim(out_dims, 1);
      
      GraphNode *in = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[(3) - (6)].id), td_in);
      GraphNode *op = add_graph_node(tensorGraph, OP_REDUCE, NULL, td_out);
      add_edge(in, op);
      
      (yyval.ast) = newNode(0, T_REDUCE, "reduce", 0, 0.0f, NULL, NULL);
    ;}
    break;


/* Line 1267 of yacc.c.  */
#line 2003 "finalAssignment.tab.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 399 "finalAssignment.y"


/*------------------------------------------------------------------
  error handler
------------------------------------------------------------------*/
int yyerror(const char *s) {
  fprintf(stderr,"%s at '%s' [%d,%d]\n",s,yytext,line,order);
  exit(1);
}

/*------------------------------------------------------------------
  evaluator & executor
------------------------------------------------------------------*/
float evalF(ARST *n) {
  if (!n) return 0.0f;
  switch (n->token) {
    case T_INTEGER: return n->valInt;
    case T_FLOAT:   return n->valFlt;
    case T_ID: {
      ARST *v = ht_search(symbolTable,n->valStr);
      return v ? v->valFlt : 0.0f;
    }
    case T_PLUSOP:  return evalF(n->left)+evalF(n->right);
    case T_MINUSOP: return evalF(n->left)-evalF(n->right);
    case T_MULTOP:  return evalF(n->left)*evalF(n->right);
    case T_DIVOP:   return (evalF(n->right)!=0.0f
                           ? evalF(n->left)/evalF(n->right)
                           : 0.0f);
    case T_MODOP:   return fmodf(evalF(n->left),evalF(n->right));
    case T_POWOP:   return potenciaConWhile(evalF(n->left),(int)evalF(n->right));
    default:        return 0.0f;
  }
}

int evalCond(ARST *n) {
  float a=evalF(n->left), b=evalF(n->right);
  switch(n->token){
    case T_LESSOP:    return a<b;
    case T_MOREOP:    return a>b;
    case T_LESSOEQOP: return a<=b;
    case T_MOREOEQOP: return a>=b;
    case T_EQSYM:     return fabsf(a-b)<1e-6f;
    default:          return 0;
  }
}

void execAST(ARST *n){
  if(!n) return;
  switch(n->token){
    case T_PROGRAM:
      execAST(n->left);
      execAST(n->right);
      break;
    case T_ASSOP:{
      float v=evalF(n->right);
      ARST *var=ht_search(symbolTable,n->valStr);
      if(var){ var->valFlt=v; var->valInt=(int)v; }
      break;
    }
    case T_DOIF:
      if(evalCond(n->left)) execAST(n->right);
      break;
    case T_DOIFELSE:{
      ARST *eb=n->right;
      if(evalCond(n->left))       execAST(eb->left);
      else                         execAST(eb->right);
      break;
    }
    case T_DOWHILE:
      do{execAST(n->right);}while(evalCond(n->left));
      break;
    case T_DOUNTIL:
      do{execAST(n->right);}while(!evalCond(n->left));
      break;
    case T_PRINT:{
      float v=evalF(n->right);
      if(v!=(int)v) printf("%f\n",v);
      else          printf("%d\n",(int)v);
      break;
    }
    case T_READ:{
      ARST *idn=n->left;
      int   x; scanf("%d",&x);
      ARST *var=ht_search(symbolTable,idn->valStr);
      if(var){ var->valInt=x; var->valFlt=x; }
      break;
    }
    default:
      execAST(n->left);
      execAST(n->right);
  }
}

float potenciaConWhile(float base,int exp){
  float r=1.0f; while(exp-->0) r*=base; return r;
}

/*------------------------------------------------------------------
  print AST by levels
------------------------------------------------------------------*/
static int height(ARST *n){
  if(!n) return 0;
  int l=height(n->left), r=height(n->right);
  return 1+(l>r?l:r);
}
static void printGivenLevel(ARST *n,int lvl){
  if(!n) return;
  if(lvl==1){
    if(n->valStr)                           printf("%s ",n->valStr);
    else if(fabsf(n->valFlt-(int)n->valFlt)>1e-6f) printf("%f ",n->valFlt);
    else                                    printf("%d ",n->valInt);
  } else {
    printGivenLevel(n->left,lvl-1);
    printGivenLevel(n->right,lvl-1);
  }
}
void printTreeLevels(ARST *root){
  int h=height(root);
  for(int i=1;i<=h;i++){
    printf("Level %d: ",i);
    printGivenLevel(root,i);
    printf("\n");
  }
}

/*------------------------------------------------------------------
  symbol‐table implementation
------------------------------------------------------------------*/
#define CAPACITY 50000

static unsigned long _hash(const char*s){
  unsigned long h=0; for(;*s;++s) h+=(unsigned char)*s;
  return h%CAPACITY;
}
static LinkedList *_alloc_list(void){
  return calloc(1,sizeof *(_alloc_list()));
}
static Ht_item *_make_item(const char*k,ARST*n){
  Ht_item *it=malloc(sizeof *it);
  it->key=strdup(k);
  it->treeNode=n;
  return it;
}
static void _collide(HashTable*t,unsigned long i,Ht_item*it){
  LinkedList *b=t->overflow_buckets[i];
  if(!b){ b=_alloc_list(); b->item=it; t->overflow_buckets[i]=b; }
  else   { LinkedList*n=_alloc_list(); n->item=it; n->next=b->next; b->next=n; }
}

HashTable *create_table(void){
  HashTable *t=malloc(sizeof *t);
  t->size=CAPACITY; t->count=0;
  t->items=calloc(CAPACITY,sizeof(Ht_item*));
  t->overflow_buckets=calloc(CAPACITY,sizeof(LinkedList*));
  return t;
}
void ht_insert(HashTable*t,char*key,ARST*node){
  unsigned long i=_hash(key);
  if(!t->items[i]){ t->items[i]=_make_item(key,node); t->count++; }
  else if(strcmp(t->items[i]->key,key)==0){ t->items[i]->treeNode=node; }
  else{ _collide(t,i,_make_item(key,node)); }
}
ARST *ht_search(HashTable*t,char*key){
  unsigned long i=_hash(key);
  Ht_item*it=t->items[i];
  if(it&&strcmp(it->key,key)==0) return it->treeNode;
  for(LinkedList*h=t->overflow_buckets[i];h;h=h->next)
    if(strcmp(h->item->key,key)==0) return h->item->treeNode;
  return NULL;
}
void print_table(HashTable*t){
  printf("\n*** SYMBOL TABLE ***\n");
  for(int i=0;i<t->size;i++){
    if(t->items[i]){
      ARST*v=t->items[i]->treeNode;
      printf("%s = ",v->valStr);
      if(v->valFlt!=(int)v->valFlt) printf("%f\n",v->valFlt);
      else                          printf("%d\n",v->valInt);
    }
    for(LinkedList*h=t->overflow_buckets[i];h;h=h->next){
      ARST*v=h->item->treeNode;
      printf("%s = ",v->valStr);
      if(v->valFlt!=(int)v->valFlt) printf("%f\n",v->valFlt);
      else                          printf("%d\n",v->valInt);
    }
  }
}
void free_table(HashTable*t){
  for(int i=0;i<t->size;i++){
    if(t->items[i]) free(t->items[i]->key),free(t->items[i]);
    LinkedList*h=t->overflow_buckets[i];
    while(h){
      free(h->item->key);
      free(h->item);
      LinkedList*n=h->next;
      free(h);
      h=n;
    }
  }
  free(t->items);
  free(t->overflow_buckets);
  free(t);
}

/*------------------------------------------------------------------
  AST node constructor
------------------------------------------------------------------*/
ARST *newNode(unsigned char type,int token,char*valStr,
              int valInt,float valFlt,ARST*L,ARST*R)
{
  ARST *n = malloc(sizeof *n);
  n->type   = type;
  n->token  = token;
  n->valStr = valStr;
  n->valInt = valInt;
  n->valFlt = valFlt;
  n->left   = L;
  n->right  = R;
  return n;
}

/*------------------------------------------------------------------
  entry point
------------------------------------------------------------------*/
int main(int argc,char**argv){
  symbolTable = create_table();    /* allocate symbol‐table up front */
  tensorGraph = NULL;                /* Initialize tensor graph */
  if(argc>1) yyin=fopen(argv[1],"r");
  yyparse();
  return 0;
}

