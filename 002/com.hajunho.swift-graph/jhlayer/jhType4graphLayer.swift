//
//  jhType4graphLayer.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 31/10/2018.
//  Copyright © 2018 hajunho.com. All rights reserved.
//

import UIKit

class jhType4graphLayer<T> : jhCommonDataLayer<T>, jhLayer_p {
    //TODO: escaping hardcoding later on
    private let xMargin : CGFloat = 45
    private let hhhWidth : CGFloat = 86400 //24h
    private let maxY : CGFloat = 400
    
    override func draw(in ctx: CGContext) {
        "".pwd(self)
        
        let isTestMode: Bool = true
        let xRatio : CGFloat = self.bounds.width / hhhWidth
        
        guard
            var ctime = (self.superScene as? jhSceneTimeLine)?.currentTime,
            var etime = (self.superScene as? jhSceneTimeLine)?.endTime
            else { return }
        
        if GS.s.logLevel.contains(.network2) {
            print("ctime in jhType1graphLayer<T>\(ctime)")
            print("ctime oh no actually etime in jhType1graphLayer<T>", etime)
        }
        
        panelID = 0 //TODO: management should be in the same class which network class related with it
        mValuesOfDatas.removeAll()
        
        guard let jhDatas = jhDataCenter.mDatas[panelID] else {
            return
        }
        
        let yRatio = self.bounds.height / maxY
        
        for man in 0..<jhDatas.d.count {
            //            mValuesOfDatas.append(jhDatas.d[man].y)
            if(GS.s.logLevel.contains(.graph2)) {
                print(jhDatas.d[man].x)
                print(jhDatas.d[man].y)
            }
            
            let x = jhDatas.d[man].x
            let y = jhDatas.d[man].y
            
            var fx = ((CGFloat(x) - CGFloat(etime.timeIntervalSince1970)) * xRatio)
            let fy = CGFloat(y) * yRatio
            
            if(GS.s.logLevel.contains(.layer)) {
                print("(fx,fy) (", fx, ")    (", fy, ")")
            }
            
            fx = fx * (1410/1500) + xMargin //TODO:
            
            if x >= CGFloat(ctime.timeIntervalSince1970) || x <= CGFloat(etime.timeIntervalSince1970) {
                continue
            } else {
                let rectangle = CGRect(x: fx-1, y: fy-8, width: 2, height: 16) //TODO:
                let clipPath = UIBezierPath(roundedRect: rectangle, cornerRadius: 3.0).cgPath
                
                ctx.addPath(clipPath)
                ctx.setFillColor(UIColor(red: 188, green: 188, blue: 188).cgColor)
                
                ctx.closePath()
                ctx.fillPath()
                
                drawPoint(ctx, fx, fy, 1, 1, thickness: 2, UIColor(red: 128, green: 128, blue: 128).cgColor)
            }
        }
        //            jhDataCenter.mDatas[panelID]!
        if isTestMode {
            testModeDrawing(in: ctx)
        }
    }
    
    func testModeDrawing(in ctx: CGContext) {
        var x : Int = 0
        var pointCloud = Array<CGPoint>()
        var fx, fy : CGFloat
        
        for man in jhDataCenter.nonNetworkData {
            mValuesOfDatas.append(man)
        }
        
        for y in mValuesOfDatas { //TODO:
            //ref:drawLine(CGFloat(x)*axisDistance + mMargin, mMargin, CGFloat(x) * axisDistance + mMargin, 10000-mMargin)
            x += 1
            fx = CGFloat(x) * xDistance
            fy = CGFloat(y) * self.bounds.height / jhDraw.ARQ + GV.s.ui_common_margin
            drawTestPoint(ctx, fx, fy, 2, 2, thickness: 2, UIColor.blue.cgColor)
            pointCloud.append(CGPoint.init(x: getXonVPanel(fx+GV.s.ui_common_margin)!, y: getYonVPanel(fy)!))
        }
        
        ctx.move(to: CGPoint.init(x: 0, y: 0))
        ctx.setStrokeColorSpace(CGColorSpaceCreateDeviceRGB())
        ctx.setStrokeColor(UIColor.blue.cgColor)
        ctx.setLineWidth(1.0)
        ctx.addLines(between: pointCloud)
        ctx.strokePath()
    }
    
    override func drawPoint(_ ctx: CGContext, _ x : CGFloat, _ y : CGFloat, _ width : CGFloat, _ height : CGFloat, thickness : CGFloat, _ color : CGColor){
        jhDraw.worldEllipse(context: ctx, x, y, width, height, thickness, color)
        //        x++GV.s.ui_common_margin
    }
    
    func drawTestPoint(_ ctx: CGContext, _ x : CGFloat, _ y : CGFloat, _ width : CGFloat, _ height : CGFloat, thickness : CGFloat, _ color : CGColor){
        //        worldEllipse(context: mContext, getX(x)!, getY(jhDraw.maxR - y)!, width, height, thickness, color)
        if GS.s.logLevel.contains(.graph) {
            print("worldEllipse(context: mContext,", getXonVPanel(x+GV.s.ui_common_margin)!, getYonVPanel(jhDraw.ARQ-y)!, width, height, thickness, color)
        }
        jhDraw.worldEllipse(context: ctx, getXonVPanel(x+GV.s.ui_common_margin)!, getYonVPanel(y)!, width, height, thickness, color)
    }
    
    func getXonVPanel(_ x: CGFloat) -> CGFloat? {
        var retX : CGFloat? = nil
        retX = x * self.bounds.width / jhDraw.ARQ
        return retX
    }
    
    func getYonVPanel(_ y: CGFloat) -> CGFloat? {
        var retY : CGFloat? = nil
        retY = y * self.bounds.width / jhDraw.ARQ
        return retY
    }
}


