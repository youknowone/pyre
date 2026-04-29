//! Port of `rpython/tool/algo/sparsemat.py`.
//!
//! Sparse matrix in row-major dictionary form.  Used by
//! `translator/backendopt/inline.measure_median_execution_cost`
//! to solve the "what fraction of paths goes through this block?"
//! linear system.
//!
//! Upstream `EPSILON = 1E-12` thresholds zero entries; entries
//! whose absolute value falls under EPSILON are dropped from the
//! row map. The Rust port mirrors the threshold verbatim.

use std::collections::HashMap;

/// `EPSILON = 1E-12` at `sparsemat.py:3`.
pub const EPSILON: f64 = 1.0e-12;

/// `class SparseMatrix` at `sparsemat.py:6-74`.
///
/// ```python
/// class SparseMatrix:
///     def __init__(self, height):
///         self.lines = [{} for row in range(height)]
/// ```
///
/// `lines[row][col] = value` stores the matrix entry. Reads of
/// missing entries return 0 (`__getitem__`). Writes of values
/// below EPSILON delete the entry instead.
#[derive(Clone, Debug)]
pub struct SparseMatrix {
    lines: Vec<HashMap<usize, f64>>,
}

impl SparseMatrix {
    /// `__init__(self, height)` (`:8-9`).
    pub fn new(height: usize) -> Self {
        let lines = (0..height).map(|_| HashMap::new()).collect();
        Self { lines }
    }

    /// `__getitem__(self, (row, col))` (`:11-12`).
    ///
    /// > return self.lines[row].get(col, 0)
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.lines
            .get(row)
            .and_then(|line| line.get(&col).copied())
            .unwrap_or(0.0)
    }

    /// `__setitem__(self, (row, col), value)` (`:14-21`).
    ///
    /// ```python
    /// def __setitem__(self, (row, col), value):
    ///     if abs(value) > EPSILON:
    ///         self.lines[row][col] = value
    ///     else:
    ///         try:
    ///             del self.lines[row][col]
    ///         except KeyError:
    ///             pass
    /// ```
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        if value.abs() > EPSILON {
            self.lines[row].insert(col, value);
        } else {
            self.lines[row].remove(&col);
        }
    }

    /// In-place mutation: `self.lines[row][col] += delta`. Used by
    /// upstream `inline.measure_median_execution_cost` at `:521-523`
    /// (`M[i, blockmap[link.target]] -= b`). Routes through
    /// [`Self::set`] so the EPSILON threshold drops the entry when
    /// the post-add value is near zero.
    pub fn add(&mut self, row: usize, col: usize, delta: f64) {
        let new_value = self.get(row, col) + delta;
        self.set(row, col, new_value);
    }

    pub fn height(&self) -> usize {
        self.lines.len()
    }

    /// `copy(self)` (`:23-27`).
    pub fn copy(&self) -> Self {
        Self {
            lines: self.lines.clone(),
        }
    }

    /// `solve(self, vector)` (`:29-74`).
    ///
    /// > Solves  'self * [x1...xn] == vector'; returns the list
    /// > [x1...xn]. Raises ValueError if no solution or
    /// > indeterminate.
    ///
    /// Returns `Err` when the input is unsolvable — upstream
    /// `max(lst)` raises `ValueError` on the empty pivot column,
    /// pyre surfaces the same condition through
    /// [`SparseMatError::NoSolution`].
    pub fn solve(&self, vector: &[f64]) -> Result<Vec<f64>, SparseMatError> {
        // Upstream `:33 vector = list(vector)`.
        let mut vector: Vec<f64> = vector.to_vec();
        // Upstream `:34 lines = [line.copy() for line in self.lines]`.
        let mut lines: Vec<HashMap<usize, f64>> = self.lines.iter().cloned().collect();
        // Upstream `:35 columns = [{} for i in range(len(vector))]`.
        let mut columns: Vec<HashMap<usize, f64>> =
            (0..vector.len()).map(|_| HashMap::new()).collect();
        // Upstream `:36-38 for i, line in enumerate(lines): for j,
        // a in line.items(): columns[j][i] = a`. Out-of-range `j`
        // upstream is an `IndexError` because `columns[j]` indexes
        // a fixed-length list — translate to `NoSolution` rather
        // than silently dropping the entry, which would otherwise
        // turn an ill-formed system into a different solvable one.
        for (i, line) in lines.iter().enumerate() {
            for (&j, &a) in line.iter() {
                if j >= columns.len() {
                    return Err(SparseMatError::ColumnOutOfRange { row: i, col: j });
                }
                columns[j].insert(i, a);
            }
        }
        // Upstream `:39 lines_left =
        // dict.fromkeys(range(len(self.lines)))`.
        let mut lines_left: std::collections::HashSet<usize> = (0..self.lines.len()).collect();
        // Upstream `:40 nrows = []`.
        let mut nrows: Vec<usize> = Vec::with_capacity(vector.len());

        for ncol in 0..vector.len() {
            // Upstream `:42-44 currentcolumn = columns[ncol]; lst =
            // [(abs(a), i) for (i, a) in currentcolumn.items() if i
            // in lines_left]`.
            let currentcolumn = columns[ncol].clone();
            let mut lst: Vec<(f64, usize)> = currentcolumn
                .iter()
                .filter(|(i, _)| lines_left.contains(i))
                .map(|(&i, &a)| (a.abs(), i))
                .collect();
            // Upstream `:45 _, nrow = max(lst)    # ValueError -> no
            // solution`.
            if lst.is_empty() {
                return Err(SparseMatError::NoSolution);
            }
            // Python's max on (abs_value, index) tuples uses
            // lexicographic ordering; ties on abs_value go to the
            // larger index. Sort matches that ordering.
            lst.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
            let &(_, nrow) = lst.last().expect("lst non-empty checked above");
            // Upstream `:46 nrows.append(nrow)`.
            nrows.push(nrow);
            // Upstream `:47 del lines_left[nrow]`.
            lines_left.remove(&nrow);
            // Upstream `:48-49 line1 = lines[nrow]; maxa =
            // line1[ncol]`.
            let maxa = *lines[nrow]
                .get(&ncol)
                .expect("pivot row contains pivot column");

            // Upstream `:50-64`. Need to iterate `lst` again — but
            // the `(_, i)` only carries the row index, so re-look up
            // a from each non-pivot row.
            // Snapshot pivot row separately so we can read `line1`
            // immutably while mutating `line2 = lines[i]`.
            let line1: HashMap<usize, f64> = lines[nrow].clone();
            for (_, i) in lst.iter() {
                let i = *i;
                if i == nrow {
                    continue;
                }
                // Upstream `:52-53 line2 = lines[i]; a =
                // line2.pop(ncol)`.
                let a = match lines[i].remove(&ncol) {
                    Some(a) => a,
                    None => continue,
                };
                // Upstream `:55 factor = a / maxa`.
                let factor = a / maxa;
                // Upstream `:56 vector[i] -= factor*vector[nrow]`.
                vector[i] -= factor * vector[nrow];
                // Upstream `:57-64 for col in line1: if col >
                // ncol: ...`.
                let cols: Vec<usize> = line1.keys().copied().collect();
                for col in cols {
                    if col > ncol {
                        let l1_col = *line1.get(&col).expect("col in line1 keys");
                        let line2_col = lines[i].get(&col).copied().unwrap_or(0.0);
                        let value = line2_col - factor * l1_col;
                        if value.abs() > EPSILON {
                            lines[i].insert(col, value);
                            columns[col].insert(i, value);
                        } else {
                            lines[i].remove(&col);
                            columns[col].remove(&i);
                        }
                    }
                }
            }
        }
        // Upstream `:65-73 back-substitution loop`.
        let mut solution: Vec<f64> = vec![0.0; vector.len()];
        for i in (0..vector.len()).rev() {
            let row = nrows[i];
            let line = &lines[row];
            let mut total = vector[row];
            for (&j, &a) in line.iter() {
                if j != i {
                    total -= a * solution[j];
                }
            }
            // Upstream `:73 solution[i] = total / line[i]`.
            let pivot = *line
                .get(&i)
                .expect("back-substitution missing pivot column entry");
            solution[i] = total / pivot;
        }
        Ok(solution)
    }
}

/// Error variant returned by [`SparseMatrix::solve`]. Upstream
/// raises `ValueError` for the indeterminate case and `IndexError`
/// for an entry whose column is outside the system width; pyre
/// distinguishes the two so callers (notably
/// `inline.measure_median_execution_cost`) can pattern-match the
/// same way the upstream `try / except` would.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SparseMatError {
    /// Upstream `:45 _, nrow = max(lst)` raises `ValueError` when
    /// `lst` is empty, signalling that the pivot column has no
    /// candidate row in `lines_left` — the system is either
    /// underdetermined or inconsistent.
    NoSolution,
    /// Upstream `:38 columns[j][i] = a` raises `IndexError` when a
    /// matrix row references a column index that exceeds the
    /// vector's length. The matrix is structurally inconsistent
    /// with the input vector width.
    ColumnOutOfRange { row: usize, col: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_height_3_yields_three_empty_rows() {
        let m = SparseMatrix::new(3);
        assert_eq!(m.height(), 3);
        assert_eq!(m.get(0, 0), 0.0);
        assert_eq!(m.get(1, 0), 0.0);
        assert_eq!(m.get(2, 0), 0.0);
    }

    #[test]
    fn set_below_epsilon_drops_entry() {
        let mut m = SparseMatrix::new(2);
        m.set(0, 0, 1.0);
        assert_eq!(m.get(0, 0), 1.0);
        // EPSILON / 2 is below threshold ⇒ entry removed.
        m.set(0, 0, EPSILON / 2.0);
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn add_routes_through_set_threshold() {
        let mut m = SparseMatrix::new(1);
        m.set(0, 0, 1.0);
        m.add(0, 0, -1.0);
        // After +(-1) total = 0 ⇒ below EPSILON ⇒ drop.
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn copy_does_not_alias_lines() {
        let mut m = SparseMatrix::new(1);
        m.set(0, 0, 1.0);
        let n = m.copy();
        m.set(0, 0, 2.0);
        // n must still see the old value.
        assert_eq!(n.get(0, 0), 1.0);
        assert_eq!(m.get(0, 0), 2.0);
    }

    #[test]
    fn solve_identity_returns_input_vector() {
        let mut m = SparseMatrix::new(3);
        m.set(0, 0, 1.0);
        m.set(1, 1, 1.0);
        m.set(2, 2, 1.0);
        let sol = m.solve(&[7.0, 8.0, 9.0]).expect("identity solvable");
        assert!((sol[0] - 7.0).abs() < EPSILON);
        assert!((sol[1] - 8.0).abs() < EPSILON);
        assert!((sol[2] - 9.0).abs() < EPSILON);
    }

    #[test]
    fn solve_2x2_simple_system() {
        // [[2, 1], [1, 3]] * x = [4, 5]   =>  x = [(7/5), (6/5)]
        let mut m = SparseMatrix::new(2);
        m.set(0, 0, 2.0);
        m.set(0, 1, 1.0);
        m.set(1, 0, 1.0);
        m.set(1, 1, 3.0);
        let sol = m.solve(&[4.0, 5.0]).expect("2x2 solvable");
        assert!((sol[0] - 7.0 / 5.0).abs() < 1e-9);
        assert!((sol[1] - 6.0 / 5.0).abs() < 1e-9);
    }

    #[test]
    fn solve_singular_matrix_returns_err() {
        // Two identical rows ⇒ singular ⇒ pivot column ends up empty
        // after first elimination.
        let mut m = SparseMatrix::new(2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 1.0);
        m.set(1, 0, 1.0);
        m.set(1, 1, 1.0);
        let result = m.solve(&[1.0, 2.0]);
        assert_eq!(result, Err(SparseMatError::NoSolution));
    }
}
