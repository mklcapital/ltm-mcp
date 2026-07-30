//! The tiny verification corpus.
//!
//! Four short documents (loan / lease / tax / insurance style) that mirror the
//! kind of content the Python server was verified against. Each document is
//! stored as a single chunk. Paths are deliberately distinct so that:
//!   - the loan document is the unambiguous top hit for an appraisal-timing
//!     query, and
//!   - the fragment `mortgage_appraisal_timeline` matches exactly one document
//!     (so `ltm_get_document` returns Found, not Ambiguous).

/// One seed document. Column names mirror `documents` in server.py's schema.
pub struct SeedDoc {
    pub source_label: &'static str,
    pub root: &'static str,
    pub relpath: &'static str,
    pub mime: &'static str,
    pub doc_type: &'static str,
    pub title: &'static str,
    pub doc_date: &'static str,
    pub text_method: &'static str,
    pub text: &'static str,
}

impl SeedDoc {
    /// Full path as `ltm_search` / `ltm_get_document` report it: `root/relpath`.
    pub fn path(&self) -> String {
        format!("{}/{}", self.root, self.relpath)
    }
}

/// The unambiguous fragment for the loan document (used by the get_document
/// fragment-lookup assertion).
pub const LOAN_FRAGMENT: &str = "mortgage_appraisal_timeline";

/// The seed corpus. Order is irrelevant; ids are assigned by the database.
pub const CORPUS: &[SeedDoc] = &[
    SeedDoc {
        source_label: "seed",
        root: "/corpus",
        relpath: "loans/mortgage_appraisal_timeline.md",
        mime: "text/markdown",
        doc_type: "loan",
        title: "Mortgage Loan Appraisal Timeline",
        doc_date: "2024-03-12",
        text_method: "seed",
        text: "Mortgage loan processing checklist. After the borrower submits a \
complete loan application and the lender issues the initial disclosures, the \
loan officer orders the property appraisal. You should order the appraisal \
early in the loan process, right after the intent to proceed is signed and the \
appraisal fee is collected, so the valuation is ready before underwriting. \
Ordering the appraisal too late delays underwriting and can push back the \
closing date. The appraiser inspects the property, compares recent sales, and \
returns an appraisal report that the underwriter uses to confirm the \
loan-to-value ratio before final loan approval and closing.",
    },
    SeedDoc {
        source_label: "seed",
        root: "/corpus",
        relpath: "leases/residential_lease_agreement.md",
        mime: "text/markdown",
        doc_type: "lease",
        title: "Residential Lease Agreement",
        doc_date: "2024-01-05",
        text_method: "seed",
        text: "Residential lease agreement between landlord and tenant. The lease \
term is twelve months beginning on the first of the month. Monthly rent is due \
on the first and is considered late after the fifth. The tenant pays a security \
deposit equal to one month's rent, refundable within thirty days of move-out \
less any deductions for damage. The tenant is responsible for utilities; the \
landlord maintains structural repairs. Pets require prior written consent and \
an additional pet deposit. Early termination requires sixty days' notice.",
    },
    SeedDoc {
        source_label: "seed",
        root: "/corpus",
        relpath: "taxes/individual_tax_return_2024.md",
        mime: "text/markdown",
        doc_type: "tax return",
        title: "Individual Income Tax Return Summary 2024",
        doc_date: "2024-04-15",
        text_method: "seed",
        text: "Individual income tax return summary for the 2024 tax year. Reports \
wages, interest income, and dividends. Itemized deductions include mortgage \
interest, state and local taxes capped at the statutory limit, and charitable \
contributions. The standard deduction is compared against itemized deductions \
to minimize taxable income. Estimated quarterly payments are reconciled against \
total tax liability to compute the refund or balance due by the April filing \
deadline.",
    },
    SeedDoc {
        source_label: "seed",
        root: "/corpus",
        relpath: "insurance/homeowners_policy.md",
        mime: "text/markdown",
        doc_type: "insurance",
        title: "Homeowners Insurance Policy",
        doc_date: "2024-02-20",
        text_method: "seed",
        text: "Homeowners insurance policy declarations. The policy provides \
dwelling coverage, personal property coverage, and personal liability \
protection. The annual premium is billed monthly through escrow. A deductible \
applies per claim before the insurer pays. Covered perils include fire, wind, \
hail, and theft; flood and earthquake are excluded and require separate \
policies. Claims are filed with the adjuster, who inspects the damage and \
issues payment up to the coverage limits after the deductible.",
    },
];
