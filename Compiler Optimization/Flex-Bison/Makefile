CC=gcc
CFLAGS=-Wall -g
FLEX=flex
BISON=bison
LDFLAGS=-lm

# Source files
LEX_SRC=finalAssignment.l
YACC_SRC=finalAssignment.y
C_SRCS=tensor_graph.c optimizer.c cuda_gen.c geometry.c
HDRS=types.h tensor_graph.h optimizer.h cuda_gen.h geometry.h

# Generated files
LEX_OUT=lex.yy.c
YACC_OUT=finalAssignment.tab.c
YACC_HEADER=finalAssignment.tab.h

# Object files
OBJS=$(LEX_OUT:.c=.o) $(YACC_OUT:.c=.o) $(C_SRCS:.c=.o)

# Target executable
TARGET=compiler

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

$(LEX_OUT): $(LEX_SRC) $(YACC_HEADER)
	$(FLEX) $(LEX_SRC)

$(YACC_OUT) $(YACC_HEADER): $(YACC_SRC)
	$(BISON) -d $(YACC_SRC)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) $(LEX_OUT) $(YACC_OUT) $(YACC_HEADER) generated_kernels.cu

test: $(TARGET)
	@echo "Testing with pruebaT51.txt:"
	./$(TARGET) pruebaT51.txt
	@echo "\nTesting with pruebaT52.txt:"
	./$(TARGET) pruebaT52.txt

