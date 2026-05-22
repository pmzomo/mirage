// SPDX-License-Identifier: Apache-2.0
//! Append-only, length-prefixed framed trace records (spec §14 storage).
use crate::TokenTrace;
use std::io::{Read, Write};

pub fn write_record<W: Write>(w: &mut W, t: &TokenTrace) -> std::io::Result<()> {
    let bytes = bincode::serialize(t).expect("trace serializes");
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(&bytes)
}

pub fn read_all<R: Read>(r: &mut R) -> std::io::Result<Vec<TokenTrace>> {
    let mut out = Vec::new();
    let mut len_buf = [0u8; 4];
    loop {
        match r.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf)?;
        out.push(bincode::deserialize(&buf).expect("trace deserializes"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_support::sample_trace;

    #[test]
    fn round_trips_multiple_records() {
        let mut buf = Vec::new();
        let a = sample_trace(1);
        let b = sample_trace(2);
        write_record(&mut buf, &a).unwrap();
        write_record(&mut buf, &b).unwrap();
        let back = read_all(&mut &buf[..]).unwrap();
        assert_eq!(back, vec![a, b]);
    }
}
