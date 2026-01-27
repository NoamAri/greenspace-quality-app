import sys

# Read the app.py file
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "# === TAB 2: BATCH EVALUATION ==="
insert_pos = None
for i, line in enumerate(lines):
    if "# === TAB 2: BATCH EVALUATION ===" in line:
        insert_pos = i
        break

if insert_pos is None:
    print("Could not find insertion point!")
    sys.exit(1)

# DINO tab content to insert
dino_tab_content = '''                    
                    with qa_tab3:
                        st.write("**🤖 DINO Feature Embeddings**")
                        st.caption("DINOv2 ViT-S/14 extracts 384-dimensional features for scene understanding")
                        
                        if dino_scene_emb is not None and dino_veg_emb is not None:
                            col_d1, col_d2 = st.columns(2)
                            
                            with col_d1:
                                st.markdown("**🏞️ Scene Embedding**")
                                dino_scene_np = dino_scene_emb.cpu().numpy().flatten()
                                st.write(f"Dimensions: 384")
                                st.write(f"L2 Norm: {np.linalg.norm(dino_scene_np):.4f}")
                                st.write(f"Mean: {dino_scene_np.mean():.4f}")
                                st.write(f"Std: {dino_scene_np.std():.4f}")
                                
                                # Top activated features
                                top_k = 3
                                top_indices = np.argsort(np.abs(dino_scene_np))[-top_k:][::-1]
                                st.write(f"**Top {top_k} features:**")
                                for idx in top_indices:
                                    st.write(f"  • Dim {idx}: {dino_scene_np[idx]:.3f}")
                            
                            with col_d2:
                                st.markdown("**🌿 Vegetation Embedding**")
                                dino_veg_np = dino_veg_emb.cpu().numpy().flatten() if torch.is_tensor(dino_veg_emb) else dino_veg_emb.flatten()
                                st.write(f"Dimensions: 384")
                                st.write(f"L2 Norm: {np.linalg.norm(dino_veg_np):.4f}")
                                st.write(f"Mean: {dino_veg_np.mean():.4f}")
                                st.write(f"Std: {dino_veg_np.std():.4f}")
                                
                                top_indices = np.argsort(np.abs(dino_veg_np))[-top_k:][::-1]
                                st.write(f"**Top {top_k} features:**")
                                for idx in top_indices:
                                    st.write(f"  • Dim {idx}: {dino_veg_np[idx]:.3f}")
                            
                            # Feature breakdown
                            st.markdown("---")
                            st.write("**📊 Complete Feature Vector (1795 dims):**")
                            st.write("- CLIP Scene: 512 dims")
                            st.write("- CLIP Vegetation: 512 dims")
                            st.write("- **DINO Scene: 384 dims** ← Self-supervised visual features")
                            st.write("- **DINO Vegetation: 384 dims** ← Crop-specific features")
                            st.write("- Color/Texture: 3 dims (green ratio, edge density, coverage)")
                        else:
                            st.warning("⚠️ DINO features not extracted for this image")

'''

# Insert the content
lines.insert(insert_pos, dino_tab_content)

# Write back
with open(sys.argv[1], 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Inserted DINO tab content at line {insert_pos}")
