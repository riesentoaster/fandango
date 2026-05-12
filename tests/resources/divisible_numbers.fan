<start> ::= '0' | <leading_digit> <digit>+
<leading_digit> ::= r'[1-9]'

where int(str(<start>)) % 2 == 0
where int(str(<start>)) % 3 == 0
where int(str(<start>)) % 5 == 0
