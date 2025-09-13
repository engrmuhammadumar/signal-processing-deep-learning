.text
.code 32

.global vector_start
.global vector_end

vector_start:
    LDR R1, =0x10000000    @ Load address of SYS_ID register
    LDR R1, [R1]           @ Load value of SYS_ID into R1

vector_end:
    .space 1024, 0

.end
