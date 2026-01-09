/* finalAssignment.y — complete parser + evaluator + symbol‐table */

%{
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
%}

/* semantic values */
%union {
  ARST  *ast;
  char  *id;
  int    ival;
  float  fval;
}

/* tokens */
%token T_COMMA T_SEMICOL T_OPENKEY T_CLOSEKEY
%token T_OPENBRACK T_CLOSEBRACK
%token T_OPENPAR T_CLOSEPAR
%token T_PLUSOP T_MULTOP T_MINUSOP T_DIVOP T_ASSOP T_EQSYM
%token T_LESSOP T_MOREOP T_LESSOEQOP T_MOREOEQOP T_MODOP T_POWOP T_COLON
%token T_READ T_PRINT T_DOIF T_DOIFELSE T_DOWHILE T_DOUNTIL
%token T_PROGRAM T_BEGIN T_END
%token T_INT T_FLT T_TENSOR
%token T_MATMUL T_TRANSPOSE T_REDUCE T_RESHAPE
%token <id>   T_ID
%token <ival> T_INTEGER
%token <fval> T_FLOAT

/* Precedence declarations */
%left T_PLUSOP T_MINUSOP
%left T_MULTOP T_DIVOP T_MODOP
%right T_POWOP
%nonassoc T_LESSOP T_MOREOP T_LESSOEQOP T_MOREOEQOP T_EQSYM

%type <ast>
  start_program prog opt_decls decl_lst decl id_list type
  stmt opt_stmts stmt_lst expr term factor expression
  tensor_decl tensor_dims tensor_op

%start start_program

%%

/*========================================================================
  Build AST, then execute + print
========================================================================*/

start_program:
    prog {
      /* (*) symbolTable has already been created in main() before yyparse() */
      rootAST = $1;
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
      
      /* Note: Don't free tensorGraph here as it may be referenced by opt_result */
      /* The graph will be cleaned up automatically when the program exits */
      if (tensorGraph && tensorGraph->num_nodes > 0) {
        /* Only free if we have nodes, and be careful about shared references */
        /* For now, skip freeing to avoid double-free issues */
      }
      free_table(symbolTable);
    }
;

prog:
    T_PROGRAM T_ID opt_decls T_BEGIN opt_stmts T_END {
      ARST *n = newNode(0, T_PROGRAM, $2, 0, 0.0f, $3, $5);
      ht_insert(symbolTable, $2, n);
      $$ = n;
    }
;

opt_decls:
    decl_lst       { $$ = $1; }
  | /* empty */    { $$ = NULL; }
;

decl_lst:
    decl T_SEMICOL decl_lst { $$ = newNode(0,0,"decl_list",0,0.0f,$1,$3); }
  | decl T_SEMICOL         { $$ = newNode(0,0,"decl",     0,0.0f,NULL,$1); }
  | decl                   { $$ = newNode(0,0,"decl",     0,0.0f,NULL,$1); }
;

decl:
    type T_COLON id_list {
      $$ = newNode(0,0,"decl",0,0.0f,$1,$3);
    }
  | tensor_decl {
      $$ = $1;
    }
;

id_list:
    T_ID T_COMMA id_list {
      /* always insert the *leaf* ID node, not the internal list node */
      ARST *leaf = newNode(1, T_ID, $1, 0, 0.0f, NULL, NULL);
      ht_insert(symbolTable, $1, leaf);
      $$ = leaf; /* propagate the leaf */
    }
  | T_ID {
      ARST *leaf = newNode(1, T_ID, $1, 0, 0.0f, NULL, NULL);
      ht_insert(symbolTable, $1, leaf);
      $$ = leaf;
    }
;

type:
    T_INT { $$ = newNode(0, T_INT,   "type_int",   0,0.0f, NULL, NULL); }
  | T_FLT { $$ = newNode(0, T_FLT,   "type_float", 0,0.0f, NULL, NULL); }
  | T_TENSOR { $$ = newNode(0, T_TENSOR, "type_tensor", 0,0.0f, NULL, NULL); }
;

tensor_decl:
    T_TENSOR T_ID T_OPENBRACK tensor_dims T_CLOSEBRACK {
      /* Create tensor graph if it doesn't exist */
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      /* Parse dimensions */
      int *dims = (int*)$4;
      int ndims = 0;
      int *temp = dims;
      while (*temp != -1) { ndims++; temp++; }
      
      TensorDim *td = create_tensor_dim(dims, ndims);
      GraphNode *node = add_graph_node(tensorGraph, OP_IDENTITY, $2, td);
      
      /* Convert to iteration space */
      IterationSpace *space = tensor_to_iteration_space(td);
      printf("Tensor %s: ", $2);
      print_iteration_space(space);
      free_iteration_space(space);
      
      /* Free the dimensions array */
      free(dims);
      
      $$ = newNode(0, T_TENSOR, $2, 0, 0.0f, NULL, NULL);
    }
;

tensor_dims:
    T_INTEGER T_COMMA tensor_dims {
      /* Build dimension list - max 10 dimensions */
      int *prev_dims = (int*)$3;
      int *dims = malloc(11 * sizeof(int)); /* Max 10 dims + sentinel */
      int i = 0;
      while (prev_dims[i] != -1 && i < 10) {
        dims[i] = prev_dims[i];
        i++;
      }
      dims[i] = $1;
      dims[i + 1] = -1;
      $$ = (ARST*)dims;
    }
  | T_INTEGER {
      int *dims = malloc(11 * sizeof(int));
      dims[0] = $1;
      dims[1] = -1;
      $$ = (ARST*)dims;
    }
;

stmt:
  T_ID T_ASSOP tensor_op {
    if (!tensorGraph) tensorGraph = create_tensor_graph();

    /* Create assignment node in graph */
    GraphNode *node = add_graph_node(tensorGraph, OP_ASSIGN, $1, NULL);

    $$ = newNode(0, T_ASSOP, $1, 0, 0.0f, NULL, NULL);
  }
  | T_ID T_ASSOP expr {
    ARST *n = newNode(0, T_ASSOP, $1, 0, 0.0f, NULL, $3);
    ht_insert(symbolTable, $1, n);
    $$ = n;
  }
  | T_DOIF T_OPENPAR expression T_CLOSEPAR
           T_OPENBRACK opt_stmts T_CLOSEBRACK {
      $$ = newNode(0, T_DOIF, "do-if", 0,0.0f, $3, $6);
    }
  | T_DOIFELSE T_OPENPAR expression T_CLOSEPAR
               T_OPENBRACK opt_stmts T_CLOSEBRACK
               T_OPENBRACK opt_stmts T_CLOSEBRACK {
      ARST *eb = newNode(0,0,"else",0,0.0f,$6,$9);
      $$ = newNode(0, T_DOIFELSE, "do-ifelse", 0,0.0f, $3, eb);
    }
  | T_DOWHILE T_OPENPAR expression T_CLOSEPAR
              T_OPENBRACK opt_stmts T_CLOSEBRACK {
      $$ = newNode(0, T_DOWHILE, "do-while", 0,0.0f, $3, $6);
    }
  | T_DOUNTIL T_OPENPAR expression T_CLOSEPAR
              T_OPENBRACK opt_stmts T_CLOSEBRACK {
      $$ = newNode(0, T_DOUNTIL, "do-until", 0,0.0f, $3, $6);
    }
  | T_PRINT expr {
      $$ = newNode(0, T_PRINT, "print", 0,0.0f, NULL, $2);
    }
  | T_READ T_ID {
      ARST *idn = newNode(1, T_ID, $2, 0,0.0f, NULL, NULL);
      $$ = newNode(0, T_READ, "read", 0,0.0f, idn, NULL);
    }
;

opt_stmts:
    stmt_lst       { $$ = $1; }
  | /* empty */    { $$ = NULL; }
;

stmt_lst:
    stmt T_SEMICOL stmt_lst   { $$ = newNode(0,0,"stmt_lst",0,0.0f,$1,$3); }
  | stmt T_SEMICOL            { $$ = $1; }
;


expr:
    expr T_PLUSOP  term { $$ = newNode(0,T_PLUSOP, "add",0,0.0f,$1,$3); }
  | expr T_MINUSOP term { $$ = newNode(0,T_MINUSOP,"sub",0,0.0f,$1,$3); }
  | term                 { $$ = $1; }
;

term:
    term T_MULTOP factor { $$ = newNode(0,T_MULTOP,"mul",0,0.0f,$1,$3); }
  | term T_DIVOP  factor { $$ = newNode(0,T_DIVOP,"div",0,0.0f,$1,$3); }
  | term T_MODOP  factor { $$ = newNode(0,T_MODOP,"mod",0,0.0f,$1,$3); }
  | term T_POWOP  factor { $$ = newNode(0,T_POWOP,"pow",0,0.0f,$1,$3); }
  | factor               { $$ = $1; }
;

factor:
    T_OPENPAR expr T_CLOSEPAR { $$ = $2; }
  | T_ID {
      ARST *n = newNode(1, T_ID, $1, 0, 0.0f, NULL, NULL);
      ht_insert(symbolTable, $1, n);
      $$ = n;
    }
  | T_FLOAT   { $$ = newNode(1, T_FLOAT, NULL, 0, $1, NULL, NULL); }
  | T_INTEGER { $$ = newNode(1, T_INTEGER, NULL, $1, 0.0f, NULL, NULL); }
;

expression:
    expr T_LESSOP    expr { $$ = newNode(0,T_LESSOP,   "lt",0,0.0f,$1,$3); }
  | expr T_MOREOP    expr { $$ = newNode(0,T_MOREOP,   "gt",0,0.0f,$1,$3); }
  | expr T_LESSOEQOP expr { $$ = newNode(0,T_LESSOEQOP,"le",0,0.0f,$1,$3); }
  | expr T_MOREOEQOP expr { $$ = newNode(0,T_MOREOEQOP,"ge",0,0.0f,$1,$3); }
  | expr T_EQSYM     expr { $$ = newNode(0,T_EQSYM,    "eq",0,0.0f,$1,$3); }
;

tensor_op:
    T_ID T_MATMUL T_ID {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      /* Create matmul operation in graph */
      int dims1[] = {100, 50}; /* Example - would get from symbol table */
      int dims2[] = {50, 200};
      TensorDim *td1 = create_tensor_dim(dims1, 2);
      TensorDim *td2 = create_tensor_dim(dims2, 2);
      int out_dims[] = {100, 200};
      TensorDim *td_out = create_tensor_dim(out_dims, 2);
      
      GraphNode *in1 = add_graph_node(tensorGraph, OP_IDENTITY, $1, td1);
      GraphNode *in2 = add_graph_node(tensorGraph, OP_IDENTITY, $3, td2);
      GraphNode *op = add_graph_node(tensorGraph, OP_MATMUL, NULL, td_out);
      add_edge(in1, op);
      add_edge(in2, op);
      
      $$ = newNode(0, T_MATMUL, "matmul", 0, 0.0f, NULL, NULL);
    }
  | T_TRANSPOSE T_OPENPAR T_ID T_CLOSEPAR {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      int dims[] = {100, 50};
      TensorDim *td_in = create_tensor_dim(dims, 2);
      int out_dims[] = {50, 100};
      TensorDim *td_out = create_tensor_dim(out_dims, 2);
      
      GraphNode *in = add_graph_node(tensorGraph, OP_IDENTITY, $3, td_in);
      GraphNode *op = add_graph_node(tensorGraph, OP_TRANSPOSE, NULL, td_out);
      add_edge(in, op);
      
      $$ = newNode(0, T_TRANSPOSE, "transpose", 0, 0.0f, NULL, NULL);
    }
  | T_REDUCE T_OPENPAR T_ID T_COMMA T_INTEGER T_CLOSEPAR {
      if (!tensorGraph) tensorGraph = create_tensor_graph();
      
      int dims[] = {100, 100};
      TensorDim *td_in = create_tensor_dim(dims, 2);
      int out_dims[] = {100};
      TensorDim *td_out = create_tensor_dim(out_dims, 1);
      
      GraphNode *in = add_graph_node(tensorGraph, OP_IDENTITY, $3, td_in);
      GraphNode *op = add_graph_node(tensorGraph, OP_REDUCE, NULL, td_out);
      add_edge(in, op);
      
      $$ = newNode(0, T_REDUCE, "reduce", 0, 0.0f, NULL, NULL);
    }
;
%%

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
      /* Check if this is a tensor operation (right side is NULL or tensor operation) */
      if (n->right == NULL || n->right->token == T_MATMUL || 
          n->right->token == T_PLUSOP || n->right->token == T_MULTOP ||
          n->right->token == T_TRANSPOSE || n->right->token == T_REDUCE) {
        /* Tensor assignment - already handled during parsing, skip evaluation */
        break;
      }
      /* Regular scalar assignment */
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
