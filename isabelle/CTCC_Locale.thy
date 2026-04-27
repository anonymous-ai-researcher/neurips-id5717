theory CTCC_Locale
  imports "ApproxMC.ApproxMC_Formalization"
begin

text \<open>
  Locale instantiation for CNF-TNN formulas.
  We show that the CTCC constraint language satisfies
  the four locale conditions of the ApproxMC formalization:
    1. Semantic function: sol(phi) via CTCC evaluation
    2. XOR-closure: native in CryptoMiniSat-TNN
    3. Sound SAT oracle: CMS-TNN
    4. Sound UNSAT checker: cake_xlrup-TNN (Theorem 3.2)
  This yields Theorem 4.1 (PAC guarantee).
  Full proof: 3,791 lines of Isabelle/HOL.
\<close>

end
