/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
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

#line 134 "finalAssignment.tab.c"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "finalAssignment.tab.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_T_COMMA = 3,                    /* T_COMMA  */
  YYSYMBOL_T_SEMICOL = 4,                  /* T_SEMICOL  */
  YYSYMBOL_T_OPENKEY = 5,                  /* T_OPENKEY  */
  YYSYMBOL_T_CLOSEKEY = 6,                 /* T_CLOSEKEY  */
  YYSYMBOL_T_OPENBRACK = 7,                /* T_OPENBRACK  */
  YYSYMBOL_T_CLOSEBRACK = 8,               /* T_CLOSEBRACK  */
  YYSYMBOL_T_OPENPAR = 9,                  /* T_OPENPAR  */
  YYSYMBOL_T_CLOSEPAR = 10,                /* T_CLOSEPAR  */
  YYSYMBOL_T_PLUSOP = 11,                  /* T_PLUSOP  */
  YYSYMBOL_T_MULTOP = 12,                  /* T_MULTOP  */
  YYSYMBOL_T_MINUSOP = 13,                 /* T_MINUSOP  */
  YYSYMBOL_T_DIVOP = 14,                   /* T_DIVOP  */
  YYSYMBOL_T_ASSOP = 15,                   /* T_ASSOP  */
  YYSYMBOL_T_EQSYM = 16,                   /* T_EQSYM  */
  YYSYMBOL_T_LESSOP = 17,                  /* T_LESSOP  */
  YYSYMBOL_T_MOREOP = 18,                  /* T_MOREOP  */
  YYSYMBOL_T_LESSOEQOP = 19,               /* T_LESSOEQOP  */
  YYSYMBOL_T_MOREOEQOP = 20,               /* T_MOREOEQOP  */
  YYSYMBOL_T_MODOP = 21,                   /* T_MODOP  */
  YYSYMBOL_T_POWOP = 22,                   /* T_POWOP  */
  YYSYMBOL_T_COLON = 23,                   /* T_COLON  */
  YYSYMBOL_T_READ = 24,                    /* T_READ  */
  YYSYMBOL_T_PRINT = 25,                   /* T_PRINT  */
  YYSYMBOL_T_DOIF = 26,                    /* T_DOIF  */
  YYSYMBOL_T_DOIFELSE = 27,                /* T_DOIFELSE  */
  YYSYMBOL_T_DOWHILE = 28,                 /* T_DOWHILE  */
  YYSYMBOL_T_DOUNTIL = 29,                 /* T_DOUNTIL  */
  YYSYMBOL_T_PROGRAM = 30,                 /* T_PROGRAM  */
  YYSYMBOL_T_BEGIN = 31,                   /* T_BEGIN  */
  YYSYMBOL_T_END = 32,                     /* T_END  */
  YYSYMBOL_T_INT = 33,                     /* T_INT  */
  YYSYMBOL_T_FLT = 34,                     /* T_FLT  */
  YYSYMBOL_T_TENSOR = 35,                  /* T_TENSOR  */
  YYSYMBOL_T_MATMUL = 36,                  /* T_MATMUL  */
  YYSYMBOL_T_TRANSPOSE = 37,               /* T_TRANSPOSE  */
  YYSYMBOL_T_REDUCE = 38,                  /* T_REDUCE  */
  YYSYMBOL_T_RESHAPE = 39,                 /* T_RESHAPE  */
  YYSYMBOL_T_ID = 40,                      /* T_ID  */
  YYSYMBOL_T_INTEGER = 41,                 /* T_INTEGER  */
  YYSYMBOL_T_FLOAT = 42,                   /* T_FLOAT  */
  YYSYMBOL_YYACCEPT = 43,                  /* $accept  */
  YYSYMBOL_start_program = 44,             /* start_program  */
  YYSYMBOL_prog = 45,                      /* prog  */
  YYSYMBOL_opt_decls = 46,                 /* opt_decls  */
  YYSYMBOL_decl_lst = 47,                  /* decl_lst  */
  YYSYMBOL_decl = 48,                      /* decl  */
  YYSYMBOL_id_list = 49,                   /* id_list  */
  YYSYMBOL_type = 50,                      /* type  */
  YYSYMBOL_tensor_decl = 51,               /* tensor_decl  */
  YYSYMBOL_tensor_dims = 52,               /* tensor_dims  */
  YYSYMBOL_stmt = 53,                      /* stmt  */
  YYSYMBOL_opt_stmts = 54,                 /* opt_stmts  */
  YYSYMBOL_stmt_lst = 55,                  /* stmt_lst  */
  YYSYMBOL_expr = 56,                      /* expr  */
  YYSYMBOL_term = 57,                      /* term  */
  YYSYMBOL_factor = 58,                    /* factor  */
  YYSYMBOL_expression = 59,                /* expression  */
  YYSYMBOL_tensor_op = 60                  /* tensor_op  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;




#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int8 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if !defined yyoverflow

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
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
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
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* !defined yyoverflow */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

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
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  122

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   297


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
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
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   102,   102,   140,   148,   149,   153,   154,   155,   159,
     162,   168,   174,   182,   183,   184,   188,   215,   228,   237,
     243,   251,   255,   261,   265,   269,   272,   279,   280,   284,
     285,   290,   291,   292,   296,   297,   298,   299,   300,   304,
     305,   310,   311,   315,   316,   317,   318,   319,   323,   342,
     356,   370,   384
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if YYDEBUG || 0
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "T_COMMA", "T_SEMICOL",
  "T_OPENKEY", "T_CLOSEKEY", "T_OPENBRACK", "T_CLOSEBRACK", "T_OPENPAR",
  "T_CLOSEPAR", "T_PLUSOP", "T_MULTOP", "T_MINUSOP", "T_DIVOP", "T_ASSOP",
  "T_EQSYM", "T_LESSOP", "T_MOREOP", "T_LESSOEQOP", "T_MOREOEQOP",
  "T_MODOP", "T_POWOP", "T_COLON", "T_READ", "T_PRINT", "T_DOIF",
  "T_DOIFELSE", "T_DOWHILE", "T_DOUNTIL", "T_PROGRAM", "T_BEGIN", "T_END",
  "T_INT", "T_FLT", "T_TENSOR", "T_MATMUL", "T_TRANSPOSE", "T_REDUCE",
  "T_RESHAPE", "T_ID", "T_INTEGER", "T_FLOAT", "$accept", "start_program",
  "prog", "opt_decls", "decl_lst", "decl", "id_list", "type",
  "tensor_decl", "tensor_dims", "stmt", "opt_stmts", "stmt_lst", "expr",
  "term", "factor", "expression", "tensor_op", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-94)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
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

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_int8 yydefact[] =
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

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -94,   -94,   -94,   -94,    89,   -94,    35,   -94,   -94,    60,
     -94,   -93,    69,   -20,    -6,   -35,    21,   -94
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
       0,     2,     3,     9,    10,    11,    31,    12,    13,    33,
      26,    27,    28,    59,    40,    41,    60,    68
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int8 yytable[] =
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

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_int8 yystos[] =
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

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr1[] =
{
       0,    43,    44,    45,    46,    46,    47,    47,    47,    48,
      48,    49,    49,    50,    50,    50,    51,    52,    52,    53,
      53,    53,    53,    53,    53,    53,    53,    54,    54,    55,
      55,    56,    56,    56,    57,    57,    57,    57,    57,    58,
      58,    58,    58,    59,    59,    59,    59,    59,    60,    60,
      60,    60,    60
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     6,     1,     0,     3,     2,     1,     3,
       1,     3,     1,     1,     1,     1,     5,     3,     1,     3,
       3,     7,    10,     7,     7,     2,     2,     1,     0,     3,
       2,     3,     3,     1,     3,     3,     3,     3,     1,     3,
       1,     1,     1,     3,     3,     3,     3,     3,     3,     3,
       3,     4,     6
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use YYerror or YYUNDEF. */
#define YYERRCODE YYUNDEF


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)




# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  yy_symbol_value_print (yyo, yykind, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp,
                 int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)]);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
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






/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep)
{
  YY_USE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = YYEMPTY; /* Cause a token to be read.  */

  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = YYEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == YYerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = YYUNDEF;
      yytoken = YYSYMBOL_YYerror;
      goto yyerrlab1;
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
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
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
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* start_program: prog  */
#line 102 "finalAssignment.y"
         {
      /* (*) symbolTable has already been created in main() before yyparse() */
      rootAST = (yyvsp[0].ast);
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
    }
#line 1289 "finalAssignment.tab.c"
    break;

  case 3: /* prog: T_PROGRAM T_ID opt_decls T_BEGIN opt_stmts T_END  */
#line 140 "finalAssignment.y"
                                                     {
      ARST *n = newNode(0, T_PROGRAM, (yyvsp[-4].id), 0, 0.0f, (yyvsp[-3].ast), (yyvsp[-1].ast));
      ht_insert(symbolTable, (yyvsp[-4].id), n);
      (yyval.ast) = n;
    }
#line 1299 "finalAssignment.tab.c"
    break;

  case 4: /* opt_decls: decl_lst  */
#line 148 "finalAssignment.y"
                   { (yyval.ast) = (yyvsp[0].ast); }
#line 1305 "finalAssignment.tab.c"
    break;

  case 5: /* opt_decls: %empty  */
#line 149 "finalAssignment.y"
                   { (yyval.ast) = NULL; }
#line 1311 "finalAssignment.tab.c"
    break;

  case 6: /* decl_lst: decl T_SEMICOL decl_lst  */
#line 153 "finalAssignment.y"
                            { (yyval.ast) = newNode(0,0,"decl_list",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1317 "finalAssignment.tab.c"
    break;

  case 7: /* decl_lst: decl T_SEMICOL  */
#line 154 "finalAssignment.y"
                           { (yyval.ast) = newNode(0,0,"decl",     0,0.0f,NULL,(yyvsp[-1].ast)); }
#line 1323 "finalAssignment.tab.c"
    break;

  case 8: /* decl_lst: decl  */
#line 155 "finalAssignment.y"
                           { (yyval.ast) = newNode(0,0,"decl",     0,0.0f,NULL,(yyvsp[0].ast)); }
#line 1329 "finalAssignment.tab.c"
    break;

  case 9: /* decl: type T_COLON id_list  */
#line 159 "finalAssignment.y"
                         {
      (yyval.ast) = newNode(0,0,"decl",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast));
    }
#line 1337 "finalAssignment.tab.c"
    break;

  case 10: /* decl: tensor_decl  */
#line 162 "finalAssignment.y"
                {
      (yyval.ast) = (yyvsp[0].ast);
    }
#line 1345 "finalAssignment.tab.c"
    break;

  case 11: /* id_list: T_ID T_COMMA id_list  */
#line 168 "finalAssignment.y"
                         {
      /* always insert the *leaf* ID node, not the internal list node */
      ARST *leaf = newNode(1, T_ID, (yyvsp[-2].id), 0, 0.0f, NULL, NULL);
      ht_insert(symbolTable, (yyvsp[-2].id), leaf);
      (yyval.ast) = leaf; /* propagate the leaf */
    }
#line 1356 "finalAssignment.tab.c"
    break;

  case 12: /* id_list: T_ID  */
#line 174 "finalAssignment.y"
         {
      ARST *leaf = newNode(1, T_ID, (yyvsp[0].id), 0, 0.0f, NULL, NULL);
      ht_insert(symbolTable, (yyvsp[0].id), leaf);
      (yyval.ast) = leaf;
    }
#line 1366 "finalAssignment.tab.c"
    break;

  case 13: /* type: T_INT  */
#line 182 "finalAssignment.y"
          { (yyval.ast) = newNode(0, T_INT,   "type_int",   0,0.0f, NULL, NULL); }
#line 1372 "finalAssignment.tab.c"
    break;

  case 14: /* type: T_FLT  */
#line 183 "finalAssignment.y"
          { (yyval.ast) = newNode(0, T_FLT,   "type_float", 0,0.0f, NULL, NULL); }
#line 1378 "finalAssignment.tab.c"
    break;

  case 15: /* type: T_TENSOR  */
#line 184 "finalAssignment.y"
             { (yyval.ast) = newNode(0, T_TENSOR, "type_tensor", 0,0.0f, NULL, NULL); }
#line 1384 "finalAssignment.tab.c"
    break;

  case 16: /* tensor_decl: T_TENSOR T_ID T_OPENBRACK tensor_dims T_CLOSEBRACK  */
#line 188 "finalAssignment.y"
                                                       {
      /* Create tensor graph if it doesn't exist */
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      /* Parse dimensions */
      int *dims = (int*)(yyvsp[-1].ast);
      int ndims = 0;
      int *temp = dims;
      while (*temp != -1) { ndims++; temp++; }
      
      TensorDim *td = create_tensor_dim(dims, ndims);
      GraphNode *node = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[-3].id), td);
      
      /* Convert to iteration space */
      IterationSpace *space = tensor_to_iteration_space(td);
      printf("Tensor %s: ", (yyvsp[-3].id));
      print_iteration_space(space);
      free_iteration_space(space);
      
      /* Free the dimensions array */
      free(dims);
      
      (yyval.ast) = newNode(0, T_TENSOR, (yyvsp[-3].id), 0, 0.0f, NULL, NULL);
    }
#line 1413 "finalAssignment.tab.c"
    break;

  case 17: /* tensor_dims: T_INTEGER T_COMMA tensor_dims  */
#line 215 "finalAssignment.y"
                                  {
      /* Build dimension list - max 10 dimensions */
      int *prev_dims = (int*)(yyvsp[0].ast);
      int *dims = malloc(11 * sizeof(int)); /* Max 10 dims + sentinel */
      int i = 0;
      while (prev_dims[i] != -1 && i < 10) {
        dims[i] = prev_dims[i];
        i++;
      }
      dims[i] = (yyvsp[-2].ival);
      dims[i + 1] = -1;
      (yyval.ast) = (ARST*)dims;
    }
#line 1431 "finalAssignment.tab.c"
    break;

  case 18: /* tensor_dims: T_INTEGER  */
#line 228 "finalAssignment.y"
              {
      int *dims = malloc(11 * sizeof(int));
      dims[0] = (yyvsp[0].ival);
      dims[1] = -1;
      (yyval.ast) = (ARST*)dims;
    }
#line 1442 "finalAssignment.tab.c"
    break;

  case 19: /* stmt: T_ID T_ASSOP expr  */
#line 237 "finalAssignment.y"
                    {
    ARST *n = newNode(0, T_ASSOP, (yyvsp[-2].id), 0, 0.0f, NULL, (yyvsp[0].ast));
    ht_insert(symbolTable, (yyvsp[-2].id), n);
    (yyval.ast) = n;
  }
#line 1452 "finalAssignment.tab.c"
    break;

  case 20: /* stmt: T_ID T_ASSOP tensor_op  */
#line 243 "finalAssignment.y"
                         {
    if (!tensorGraph) tensorGraph = create_tensor_graph();

    /* Create assignment node in graph */
    GraphNode *node = add_graph_node(tensorGraph, OP_ASSIGN, (yyvsp[-2].id), NULL);

    (yyval.ast) = newNode(0, T_ASSOP, (yyvsp[-2].id), 0, 0.0f, NULL, NULL);
  }
#line 1465 "finalAssignment.tab.c"
    break;

  case 21: /* stmt: T_DOIF T_OPENPAR expression T_CLOSEPAR T_OPENBRACK opt_stmts T_CLOSEBRACK  */
#line 252 "finalAssignment.y"
                                              {
      (yyval.ast) = newNode(0, T_DOIF, "do-if", 0,0.0f, (yyvsp[-4].ast), (yyvsp[-1].ast));
    }
#line 1473 "finalAssignment.tab.c"
    break;

  case 22: /* stmt: T_DOIFELSE T_OPENPAR expression T_CLOSEPAR T_OPENBRACK opt_stmts T_CLOSEBRACK T_OPENBRACK opt_stmts T_CLOSEBRACK  */
#line 257 "finalAssignment.y"
                                                  {
      ARST *eb = newNode(0,0,"else",0,0.0f,(yyvsp[-4].ast),(yyvsp[-1].ast));
      (yyval.ast) = newNode(0, T_DOIFELSE, "do-ifelse", 0,0.0f, (yyvsp[-7].ast), eb);
    }
#line 1482 "finalAssignment.tab.c"
    break;

  case 23: /* stmt: T_DOWHILE T_OPENPAR expression T_CLOSEPAR T_OPENBRACK opt_stmts T_CLOSEBRACK  */
#line 262 "finalAssignment.y"
                                                 {
      (yyval.ast) = newNode(0, T_DOWHILE, "do-while", 0,0.0f, (yyvsp[-4].ast), (yyvsp[-1].ast));
    }
#line 1490 "finalAssignment.tab.c"
    break;

  case 24: /* stmt: T_DOUNTIL T_OPENPAR expression T_CLOSEPAR T_OPENBRACK opt_stmts T_CLOSEBRACK  */
#line 266 "finalAssignment.y"
                                                 {
      (yyval.ast) = newNode(0, T_DOUNTIL, "do-until", 0,0.0f, (yyvsp[-4].ast), (yyvsp[-1].ast));
    }
#line 1498 "finalAssignment.tab.c"
    break;

  case 25: /* stmt: T_PRINT expr  */
#line 269 "finalAssignment.y"
                 {
      (yyval.ast) = newNode(0, T_PRINT, "print", 0,0.0f, NULL, (yyvsp[0].ast));
    }
#line 1506 "finalAssignment.tab.c"
    break;

  case 26: /* stmt: T_READ T_ID  */
#line 272 "finalAssignment.y"
                {
      ARST *idn = newNode(1, T_ID, (yyvsp[0].id), 0,0.0f, NULL, NULL);
      (yyval.ast) = newNode(0, T_READ, "read", 0,0.0f, idn, NULL);
    }
#line 1515 "finalAssignment.tab.c"
    break;

  case 27: /* opt_stmts: stmt_lst  */
#line 279 "finalAssignment.y"
                   { (yyval.ast) = (yyvsp[0].ast); }
#line 1521 "finalAssignment.tab.c"
    break;

  case 28: /* opt_stmts: %empty  */
#line 280 "finalAssignment.y"
                   { (yyval.ast) = NULL; }
#line 1527 "finalAssignment.tab.c"
    break;

  case 29: /* stmt_lst: stmt T_SEMICOL stmt_lst  */
#line 284 "finalAssignment.y"
                              { (yyval.ast) = newNode(0,0,"stmt_lst",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1533 "finalAssignment.tab.c"
    break;

  case 30: /* stmt_lst: stmt T_SEMICOL  */
#line 285 "finalAssignment.y"
                              { (yyval.ast) = (yyvsp[-1].ast); }
#line 1539 "finalAssignment.tab.c"
    break;

  case 31: /* expr: expr T_PLUSOP term  */
#line 290 "finalAssignment.y"
                        { (yyval.ast) = newNode(0,T_PLUSOP, "add",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1545 "finalAssignment.tab.c"
    break;

  case 32: /* expr: expr T_MINUSOP term  */
#line 291 "finalAssignment.y"
                        { (yyval.ast) = newNode(0,T_MINUSOP,"sub",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1551 "finalAssignment.tab.c"
    break;

  case 33: /* expr: term  */
#line 292 "finalAssignment.y"
                         { (yyval.ast) = (yyvsp[0].ast); }
#line 1557 "finalAssignment.tab.c"
    break;

  case 34: /* term: term T_MULTOP factor  */
#line 296 "finalAssignment.y"
                         { (yyval.ast) = newNode(0,T_MULTOP,"mul",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1563 "finalAssignment.tab.c"
    break;

  case 35: /* term: term T_DIVOP factor  */
#line 297 "finalAssignment.y"
                         { (yyval.ast) = newNode(0,T_DIVOP,"div",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1569 "finalAssignment.tab.c"
    break;

  case 36: /* term: term T_MODOP factor  */
#line 298 "finalAssignment.y"
                         { (yyval.ast) = newNode(0,T_MODOP,"mod",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1575 "finalAssignment.tab.c"
    break;

  case 37: /* term: term T_POWOP factor  */
#line 299 "finalAssignment.y"
                         { (yyval.ast) = newNode(0,T_POWOP,"pow",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1581 "finalAssignment.tab.c"
    break;

  case 38: /* term: factor  */
#line 300 "finalAssignment.y"
                         { (yyval.ast) = (yyvsp[0].ast); }
#line 1587 "finalAssignment.tab.c"
    break;

  case 39: /* factor: T_OPENPAR expr T_CLOSEPAR  */
#line 304 "finalAssignment.y"
                              { (yyval.ast) = (yyvsp[-1].ast); }
#line 1593 "finalAssignment.tab.c"
    break;

  case 40: /* factor: T_ID  */
#line 305 "finalAssignment.y"
         {
      ARST *n = newNode(1, T_ID, (yyvsp[0].id), 0, 0.0f, NULL, NULL);
      ht_insert(symbolTable, (yyvsp[0].id), n);
      (yyval.ast) = n;
    }
#line 1603 "finalAssignment.tab.c"
    break;

  case 41: /* factor: T_FLOAT  */
#line 310 "finalAssignment.y"
              { (yyval.ast) = newNode(1, T_FLOAT, NULL, 0, (yyvsp[0].fval), NULL, NULL); }
#line 1609 "finalAssignment.tab.c"
    break;

  case 42: /* factor: T_INTEGER  */
#line 311 "finalAssignment.y"
              { (yyval.ast) = newNode(1, T_INTEGER, NULL, (yyvsp[0].ival), 0.0f, NULL, NULL); }
#line 1615 "finalAssignment.tab.c"
    break;

  case 43: /* expression: expr T_LESSOP expr  */
#line 315 "finalAssignment.y"
                          { (yyval.ast) = newNode(0,T_LESSOP,   "lt",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1621 "finalAssignment.tab.c"
    break;

  case 44: /* expression: expr T_MOREOP expr  */
#line 316 "finalAssignment.y"
                          { (yyval.ast) = newNode(0,T_MOREOP,   "gt",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1627 "finalAssignment.tab.c"
    break;

  case 45: /* expression: expr T_LESSOEQOP expr  */
#line 317 "finalAssignment.y"
                          { (yyval.ast) = newNode(0,T_LESSOEQOP,"le",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1633 "finalAssignment.tab.c"
    break;

  case 46: /* expression: expr T_MOREOEQOP expr  */
#line 318 "finalAssignment.y"
                          { (yyval.ast) = newNode(0,T_MOREOEQOP,"ge",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1639 "finalAssignment.tab.c"
    break;

  case 47: /* expression: expr T_EQSYM expr  */
#line 319 "finalAssignment.y"
                          { (yyval.ast) = newNode(0,T_EQSYM,    "eq",0,0.0f,(yyvsp[-2].ast),(yyvsp[0].ast)); }
#line 1645 "finalAssignment.tab.c"
    break;

  case 48: /* tensor_op: T_ID T_MATMUL T_ID  */
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
      
      GraphNode *in1 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[-2].id), td1);
      GraphNode *in2 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[0].id), td2);
      GraphNode *op = add_graph_node(tensorGraph, OP_MATMUL, NULL, td_out);
      add_edge(in1, op);
      add_edge(in2, op);
      
      (yyval.ast) = newNode(0, T_MATMUL, "matmul", 0, 0.0f, NULL, NULL);
    }
#line 1669 "finalAssignment.tab.c"
    break;

  case 49: /* tensor_op: T_ID T_PLUSOP T_ID  */
#line 342 "finalAssignment.y"
                       {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      int dims[] = {100, 100};
      TensorDim *td = create_tensor_dim(dims, 2);
      
      GraphNode *in1 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[-2].id), td);
      GraphNode *in2 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[0].id), td);
      GraphNode *op = add_graph_node(tensorGraph, OP_ADD, NULL, td);
      add_edge(in1, op);
      add_edge(in2, op);
      
      (yyval.ast) = newNode(0, T_PLUSOP, "tensor_add", 0, 0.0f, NULL, NULL);
    }
#line 1688 "finalAssignment.tab.c"
    break;

  case 50: /* tensor_op: T_ID T_MULTOP T_ID  */
#line 356 "finalAssignment.y"
                       {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      int dims[] = {100, 100};
      TensorDim *td = create_tensor_dim(dims, 2);
      
      GraphNode *in1 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[-2].id), td);
      GraphNode *in2 = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[0].id), td);
      GraphNode *op = add_graph_node(tensorGraph, OP_MUL, NULL, td);
      add_edge(in1, op);
      add_edge(in2, op);
      
      (yyval.ast) = newNode(0, T_MULTOP, "tensor_mul", 0, 0.0f, NULL, NULL);
    }
#line 1707 "finalAssignment.tab.c"
    break;

  case 51: /* tensor_op: T_TRANSPOSE T_OPENPAR T_ID T_CLOSEPAR  */
#line 370 "finalAssignment.y"
                                          {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      int dims[] = {100, 50};
      TensorDim *td_in = create_tensor_dim(dims, 2);
      int out_dims[] = {50, 100};
      TensorDim *td_out = create_tensor_dim(out_dims, 2);
      
      GraphNode *in = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[-1].id), td_in);
      GraphNode *op = add_graph_node(tensorGraph, OP_TRANSPOSE, NULL, td_out);
      add_edge(in, op);
      
      (yyval.ast) = newNode(0, T_TRANSPOSE, "transpose", 0, 0.0f, NULL, NULL);
    }
#line 1726 "finalAssignment.tab.c"
    break;

  case 52: /* tensor_op: T_REDUCE T_OPENPAR T_ID T_COMMA T_INTEGER T_CLOSEPAR  */
#line 384 "finalAssignment.y"
                                                         {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      int dims[] = {100, 100};
      TensorDim *td_in = create_tensor_dim(dims, 2);
      int out_dims[] = {100};
      TensorDim *td_out = create_tensor_dim(out_dims, 1);
      
      GraphNode *in = add_graph_node(tensorGraph, OP_IDENTITY, (yyvsp[-3].id), td_in);
      GraphNode *op = add_graph_node(tensorGraph, OP_REDUCE, NULL, td_out);
      add_edge(in, op);
      
      (yyval.ast) = newNode(0, T_REDUCE, "reduce", 0, 0.0f, NULL, NULL);
    }
#line 1745 "finalAssignment.tab.c"
    break;


#line 1749 "finalAssignment.tab.c"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      yyerror (YY_("syntax error"));
    }

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
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

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
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
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
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
                  YY_ACCESSING_SYMBOL (yystate), yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif

  return yyresult;
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
