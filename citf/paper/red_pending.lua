function Str(el)
  if el.text:find('待补',1,true) then
    local out={}
    local start=1
    while true do
      local a,b=el.text:find('待补',start,true)
      if not a then
        if start<=#el.text then table.insert(out,pandoc.Str(el.text:sub(start))) end
        break
      end
      if a>start then table.insert(out,pandoc.Str(el.text:sub(start,a-1))) end
      table.insert(out,pandoc.RawInline('latex','\\textcolor{pending}{\\textbf{待补}}'))
      start=b+1
    end
    return out
  end
end
