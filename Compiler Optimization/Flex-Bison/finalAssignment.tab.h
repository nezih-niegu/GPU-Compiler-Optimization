/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_YY_FINALASSIGNMENT_TAB_H_INCLUDED
# define YY_YY_FINALASSIGNMENT_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    T_COMMA = 258,                 /* T_COMMA  */
    T_SEMICOL = 259,               /* T_SEMICOL  */
    T_OPENKEY = 260,               /* T_OPENKEY  */
    T_CLOSEKEY = 261,              /* T_CLOSEKEY  */
    T_OPENBRACK = 262,             /* T_OPENBRACK  */
    T_CLOSEBRACK = 263,            /* T_CLOSEBRACK  */
    T_OPENPAR = 264,               /* T_OPENPAR  */
    T_CLOSEPAR = 265,              /* T_CLOSEPAR  */
    T_PLUSOP = 266,                /* T_PLUSOP  */
    T_MULTOP = 267,                /* T_MULTOP  */
    T_MINUSOP = 268,               /* T_MINUSOP  */
    T_DIVOP = 269,                 /* T_DIVOP  */
    T_ASSOP = 270,                 /* T_ASSOP  */
    T_EQSYM = 271,                 /* T_EQSYM  */
    T_LESSOP = 272,                /* T_LESSOP  */
    T_MOREOP = 273,                /* T_MOREOP  */
    T_LESSOEQOP = 274,             /* T_LESSOEQOP  */
    T_MOREOEQOP = 275,             /* T_MOREOEQOP  */
    T_MODOP = 276,                 /* T_MODOP  */
    T_POWOP = 277,                 /* T_POWOP  */
    T_COLON = 278,                 /* T_COLON  */
    T_READ = 279,                  /* T_READ  */
    T_PRINT = 280,                 /* T_PRINT  */
    T_DOIF = 281,                  /* T_DOIF  */
    T_DOIFELSE = 282,              /* T_DOIFELSE  */
    T_DOWHILE = 283,               /* T_DOWHILE  */
    T_DOUNTIL = 284,               /* T_DOUNTIL  */
    T_PROGRAM = 285,               /* T_PROGRAM  */
    T_BEGIN = 286,                 /* T_BEGIN  */
    T_END = 287,                   /* T_END  */
    T_INT = 288,                   /* T_INT  */
    T_FLT = 289,                   /* T_FLT  */
    T_TENSOR = 290,                /* T_TENSOR  */
    T_MATMUL = 291,                /* T_MATMUL  */
    T_TRANSPOSE = 292,             /* T_TRANSPOSE  */
    T_REDUCE = 293,                /* T_REDUCE  */
    T_RESHAPE = 294,               /* T_RESHAPE  */
    T_ID = 295,                    /* T_ID  */
    T_INTEGER = 296,               /* T_INTEGER  */
    T_FLOAT = 297                  /* T_FLOAT  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 66 "finalAssignment.y"

  ARST  *ast;
  char  *id;
  int    ival;
  float  fval;

#line 113 "finalAssignment.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_FINALASSIGNMENT_TAB_H_INCLUDED  */
