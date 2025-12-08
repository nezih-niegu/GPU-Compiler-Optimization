/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

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




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 66 "finalAssignment.y"
{
  ARST  *ast;
  char  *id;
  int    ival;
  float  fval;
}
/* Line 1529 of yacc.c.  */
#line 140 "finalAssignment.tab.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

