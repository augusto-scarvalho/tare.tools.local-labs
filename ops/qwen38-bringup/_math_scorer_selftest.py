import re, sympy
from sympy.parsing.sympy_parser import parse_expr

def norm(s):
    if s is None: return None
    s=str(s)
    s=s.replace('\\left','').replace('\\right','')
    s=re.sub(r'\\text\{[^}]*\}','',s)
    s=re.sub(r'\\!|\\,|\\;|\\ ','',s)
    s=s.replace('\\dfrac','\\frac').replace('\\tfrac','\\frac')
    s=s.replace('$','').replace('\\%','').replace('%','')
    s=s.replace('\\$','').replace(' ','')
    s=s.replace('^{\\circ}','').replace('^\\circ','')
    return s.strip()

def to_sym(s):
    if s is None: return None
    s=norm(s)
    s=re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}',r'((\1)/(\2))',s)
    s=re.sub(r'\\sqrt\{([^{}]+)\}',r'sqrt(\1)',s)
    s=re.sub(r'\\sqrt(\w)',r'sqrt(\1)',s)
    s=s.replace('\\pi','pi').replace('\\cdot','*').replace('\\times','*')
    s=s.replace('{','(').replace('}',')')
    s=re.sub(r'\\[a-zA-Z]+','',s)
    s=s.replace('^','**')
    try: return sympy.simplify(parse_expr(s,evaluate=True))
    except Exception: return None

def equiv(pred,gold):
    if pred is None: return False
    if norm(pred)==norm(gold): return True
    a,b=to_sym(pred),to_sym(gold)
    if a is not None and b is not None:
        try: return sympy.simplify(a-b)==0
        except Exception: return False
    return False

tests=[
 (r"\frac{1}{2}", r"\frac{1}{2}", True),
 (r"\dfrac{1}{2}", r"\frac{1}{2}", True),
 (r"0.5", r"\frac{1}{2}", True),
 (r"\frac{\pi}{2}", r"\frac{\pi}{2}", True),
 (r"2\sqrt{3}", r"2\sqrt{3}", True),
 (r"7", r"7", True),
 (r"7", r"8", False),
 (r"\left(3,\frac{\pi}{2}\right)", r"\left( 3, \frac{\pi}{2} \right)", True),
 (r"-\frac{3}{2}", r"-1.5", True),
 (r"\frac{\sqrt{2}}{2}", r"\frac{1}{\sqrt{2}}", True),
 (r"x=5", r"5", False),
]
ok=0
for p,g,exp in tests:
    r=equiv(p,g); ok+=(r==exp)
    print(f"{'ok ' if r==exp else 'FAIL'} equiv({p!r},{g!r})={r} want {exp}")
print(f"\nself-test: {ok}/{len(tests)} passed")
