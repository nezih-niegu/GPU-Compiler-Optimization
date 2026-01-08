%{
#include "types.h"
#include "finalAssignment.tab.h"
extern int line, order;
extern char *yytext;
%}

WS    [ \t]+
NL    (\r?\n)
ID    [A-Za-z][A-Za-z0-9_]*
INT   [0-9]+
FLT   ([0-9]+[.][0-9]*)|([.][0-9]+)

%%

"program"      { return T_PROGRAM; }
"begin"        { return T_BEGIN;   }
"end"          { return T_END;     }

"int"          { return T_INT;     }
"float"        { return T_FLT;     }
"tensor"       { return T_TENSOR;  }

"do-ifelse"    { return T_DOIFELSE; }
"do-if"        { return T_DOIF;     }
"do-while"     { return T_DOWHILE;  }
"do-until"     { return T_DOUNTIL;  }

"read"         { return T_READ;    }
"print"        { return T_PRINT;   }

","            { return T_COMMA;   }
";"            { return T_SEMICOL; }
":"            { return T_COLON;   }

"{"            { return T_OPENKEY; }
"}"            { return T_CLOSEKEY;}

"["            { return T_OPENBRACK; }
"]"            { return T_CLOSEBRACK;}

"("            { return T_OPENPAR; }
")"            { return T_CLOSEPAR;}

":="           { return T_ASSOP;   }
"="            { return T_EQSYM;   }
"<="|"=<"      { return T_LESSOEQOP; }
">="|"=> "     { return T_MOREOEQOP; }
"<"            { return T_LESSOP;  }
">"            { return T_MOREOP;  }

"+"            { return T_PLUSOP;  }
"-"            { return T_MINUSOP; }
"*"            { return T_MULTOP;  }
"/"            { return T_DIVOP;   }
"%"            { return T_MODOP;   }
"^"            { return T_POWOP;   }
"@matmul"      { return T_MATMUL;  }
"@transpose"   { return T_TRANSPOSE; }
"@reduce"      { return T_REDUCE;  }
"@reshape"     { return T_RESHAPE; }

{FLT}          { yylval.fval = atof(yytext); return T_FLOAT; }
{INT}          { yylval.ival = atoi(yytext);  return T_INTEGER; }
{ID}           { yylval.id   = strdup(yytext); return T_ID; }

{WS}           { order++; }
{NL}           { line++; order=1; }

<<EOF>>        { return 0; }   /* EOF → end parse */
.              { fprintf(stderr,"Unknown '%s' at [%d,%d]\n",yytext,line,order); exit(1); }
%%

int yywrap(void) { return 1; }
