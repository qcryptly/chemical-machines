class Profile {
  constructor(pool) {
    this.pool = pool;
  }

  async create({ name, email }) {
    const result = await this.pool.query(
      `INSERT INTO profiles (name, email)
       VALUES ($1, $2)
       RETURNING *`,
      [name, email]
    );
    return result.rows[0];
  }

  async findAll() {
    const result = await this.pool.query(
      `SELECT id, name, email, ssh_key_fingerprint, created_at, updated_at
       FROM profiles ORDER BY name`
    );
    return result.rows;
  }

  async findById(id) {
    const result = await this.pool.query(
      `SELECT * FROM profiles WHERE id = $1`,
      [id]
    );
    return result.rows[0] || null;
  }

  async findByEmail(email) {
    const result = await this.pool.query(
      `SELECT * FROM profiles WHERE email = $1`,
      [email]
    );
    return result.rows[0] || null;
  }

  async getActive() {
    // For now, just return the first profile as "active"
    // Could be extended to support multiple profiles with an active flag
    const result = await this.pool.query(
      `SELECT * FROM profiles ORDER BY id LIMIT 1`
    );
    return result.rows[0] || null;
  }

  async update(id, fields) {
    const allowedFields = ['name', 'email', 'ssh_public_key', 'ssh_private_key', 'ssh_key_fingerprint'];
    const updates = [];
    const values = [];
    let paramIndex = 1;

    for (const [key, value] of Object.entries(fields)) {
      if (allowedFields.includes(key)) {
        updates.push(`${key} = $${paramIndex}`);
        values.push(value);
        paramIndex++;
      }
    }

    if (updates.length === 0) {
      return this.findById(id);
    }

    updates.push(`updated_at = CURRENT_TIMESTAMP`);
    values.push(id);

    const result = await this.pool.query(
      `UPDATE profiles
       SET ${updates.join(', ')}
       WHERE id = $${paramIndex}
       RETURNING *`,
      values
    );
    return result.rows[0] || null;
  }

  async setSSHKeys(id, { publicKey, privateKey, fingerprint }) {
    return this.update(id, {
      ssh_public_key: publicKey,
      ssh_private_key: privateKey,
      ssh_key_fingerprint: fingerprint
    });
  }

  async getPublicKey(id) {
    const result = await this.pool.query(
      `SELECT ssh_public_key FROM profiles WHERE id = $1`,
      [id]
    );
    return result.rows[0]?.ssh_public_key || null;
  }

  async delete(id) {
    const result = await this.pool.query(
      `DELETE FROM profiles WHERE id = $1 RETURNING *`,
      [id]
    );
    return result.rows[0] || null;
  }
}

module.exports = Profile;
