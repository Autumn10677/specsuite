document.addEventListener("DOMContentLoaded", () => {

  // Finds elements in the DOM that have been flagged for having a "toggled dropdown"
  document.querySelectorAll('div[class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs celltag_hide_code_block"]').forEach(cell => {

    // Verifies that the cell has no id (only applies to outermost wrapper)
    const id = cell.id;
    if (!(cell.id === "")) return;

    // Creates a wrapper for the code block so we can control its behavior & appearance
    const wrapper = document.createElement("div");
    wrapper.classList.add("hide-input-wrapper");

    // Creates the toggle button
    const button = document.createElement("button");
    button.classList.add("hide-input-toggle");
    button.textContent = "Show Setup Code";

    // Moves the identified code block into our dropdown wrapper
    const parent = cell.parentNode;
    parent.insertBefore(wrapper, cell);
    wrapper.appendChild(button);
    wrapper.appendChild(cell);

    // Start hidden
    cell.style.display = "none";

    // Allows the user to toggle the display using an event listener
    button.addEventListener("click", () => {
      const visible = cell.style.display !== "none";
      if (visible) {
        cell.style.display = "none";
        button.textContent = "Show Setup Code";
      } else {
        cell.style.display = "";
        button.textContent = "Hide Setup Code";
      }
    });
    
  });
});
